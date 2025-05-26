import streamlit as st
import pandas as pd
import numpy as np
import joblib
import ast
import sys
import os
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

@st.cache_data
def load_data():
    """Load the movie dataset with NLP features"""
    try:
        # Try to load the dataset with NLP features first
        try:
            movies = pd.read_csv('../data/movies_with_nlp.csv')
        except:
            # Fallback to original dataset
            movies = pd.read_csv('../data/movies_dataset.csv')
            
        # Convert string columns back to lists
        for col in ['genres', 'top_cast', 'keywords']:
            if col in movies.columns:
                movies[col] = movies[col].apply(
                    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
                )
        return movies
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_resource
def load_models():
    """Load all trained models"""
    models = {}
    try:
        models['kmeans'] = joblib.load('../models/kmeans_model.joblib')
        models['meanshift'] = joblib.load('../models/meanshift_model.joblib')
        models['hierarchical'] = joblib.load('../models/agg_model.joblib')
        models['random_forest'] = joblib.load('../models/random_forest.joblib')
        models['tfidf'] = joblib.load('../models/tfidf_vectorizer.joblib')
        return models
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return {}

def preprocess_for_clustering(movies, movie_idx):
    # Limiting, encoding, and scaling features same as in training 
    all_keywords = pd.Series([kw for kws in movies['keywords'] for kw in kws])
    top_keywords = all_keywords.value_counts().head(100).index
    movies_processed = movies.copy()
    movies_processed['keywords'] = movies_processed['keywords'].apply(
        lambda kws: [kw for kw in kws if kw in top_keywords]
    )
    
    all_actors = pd.Series([actor for cast in movies['top_cast'] for actor in cast])
    top_actors = all_actors.value_counts().head(100).index
    movies_processed['top_cast'] = movies_processed['top_cast'].apply(
        lambda cast: [actor for actor in cast if actor in top_actors]
    )
    
    top_directors = movies['director'].value_counts().head(50).index
    movies_processed['director'] = movies_processed['director'].apply(
        lambda d: d if d in top_directors else 'Other'
    )
    
    # Encoding features
    mlb_genres = MultiLabelBinarizer()
    genres_encoded = mlb_genres.fit_transform(movies_processed['genres'])
    
    mlb_keywords = MultiLabelBinarizer()
    keywords_encoded = mlb_keywords.fit_transform(movies_processed['keywords'])
    
    mlb_cast = MultiLabelBinarizer()
    cast_encoded = mlb_cast.fit_transform(movies_processed['top_cast'])
    
    director_encoded = pd.get_dummies(movies_processed['director']).values
    language_encoded = pd.get_dummies(movies_processed['original_language']).values
    
    # Scaling numerical features
    num_features = ['budget', 'revenue', 'runtime', 'profit', 'vote_average', 'vote_count', 'release_year', 'decade']
    available_num_features = [col for col in num_features if col in movies_processed.columns]
    
    scaler = StandardScaler()
    num_encoded = scaler.fit_transform(movies_processed[available_num_features])
    
    X = np.hstack([
        genres_encoded,
        keywords_encoded,
        cast_encoded,
        director_encoded,
        language_encoded,
        num_encoded
    ])
    
    return X

def get_cluster_recommendations(movies, models, selected_movie, method='kmeans'):
    try:
        movie_idx = movies[movies['title'] == selected_movie].index[0]
        X = preprocess_for_clustering(movies, movie_idx)
        
        if method == 'kmeans':
            cluster = models['kmeans'].predict(X[movie_idx:movie_idx+1])[0]
            cluster_movies = movies[models['kmeans'].predict(X) == cluster]
        elif method == 'meanshift':
            cluster = models['meanshift'].predict(X[movie_idx:movie_idx+1])[0]
            cluster_movies = movies[models['meanshift'].predict(X) == cluster]
        elif method == 'hierarchical':
            cluster = models['hierarchical'].fit_predict(X)[movie_idx]
            cluster_movies = movies[models['hierarchical'].fit_predict(X) == cluster]
        
        # Remove the selected movie and return top recommendations
        recommendations = cluster_movies[cluster_movies['title'] != selected_movie]
        return recommendations.sort_values('vote_average', ascending=False).head(10)
    
    except Exception as e:
        st.error(f"Error in clustering recommendations: {e}")
        return pd.DataFrame()

def get_nlp_recommendations(movies, models, selected_movie, n=10):
    try:
        if 'nlp' not in movies.columns:
            st.warning("NLP features not available in dataset")
            return pd.DataFrame()
        
        # Create TF-IDF matrix
        tfidf_matrix = models['tfidf'].transform(movies['nlp'])
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        # Get movie index
        indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()
        
        if selected_movie not in indices:
            st.error(f"Movie '{selected_movie}' not found in dataset")
            return pd.DataFrame()
        
        idx = indices[selected_movie]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:n+1]  # Skip the first one (it's the movie itself)
        
        movie_indices = [i[0] for i in sim_scores]
        scores = [i[1] for i in sim_scores]
        
        result = movies[['title', 'genres', 'vote_average', 'release_year', 'director']].iloc[movie_indices].copy()
        result['similarity_score'] = scores
        
        return result
    
    except Exception as e:
        st.error(f"Error in NLP recommendations: {e}")
        return pd.DataFrame()
    
@st.cache_resource
def load_preprocessing_objects():
    """Load preprocessing objects for Random Forest"""
    try:
        return joblib.load('../models/rf_preprocessing.joblib')
    except Exception as e:
        st.error(f"Error loading preprocessing objects: {e}")
        return {}
    
def preprocess_movie_for_prediction(movie_data, preprocessing_objects):
    """Preprocess a single movie for Random Forest prediction"""
    try:
        # Extract preprocessing objects
        mlb_genres = preprocessing_objects['mlb_genres']
        mlb_keywords = preprocessing_objects['mlb_keywords']
        mlb_cast = preprocessing_objects['mlb_cast']
        scaler = preprocessing_objects['scaler']
        top_keywords = preprocessing_objects['top_keywords']
        top_actors = preprocessing_objects['top_actors']
        top_directors = preprocessing_objects['top_directors']
        director_categories = preprocessing_objects['director_categories']
        language_categories = preprocessing_objects['language_categories']
        num_features = preprocessing_objects['num_features']
        
        # Process features exactly as in training
        # Filter keywords and cast
        filtered_keywords = [kw for kw in movie_data['keywords'] if kw in top_keywords]
        filtered_cast = [actor for actor in movie_data['top_cast'] if actor in top_actors]
        filtered_director = movie_data['director'] if movie_data['director'] in top_directors else 'Other'
        
        # Encode features
        genres_encoded = mlb_genres.transform([movie_data['genres']])
        keywords_encoded = mlb_keywords.transform([filtered_keywords])
        cast_encoded = mlb_cast.transform([filtered_cast])
        
        # Director encoding
        director_encoded = np.zeros((1, len(director_categories)))
        if filtered_director in director_categories:
            director_idx = list(director_categories).index(filtered_director)
            director_encoded[0, director_idx] = 1
        
        # Language encoding
        language_encoded = np.zeros((1, len(language_categories)))
        if movie_data['original_language'] in language_categories:
            lang_idx = list(language_categories).index(movie_data['original_language'])
            language_encoded[0, lang_idx] = 1
        
        # Numerical features
        num_data = []
        for feature in num_features:
            if feature in movie_data and pd.notna(movie_data[feature]):
                num_data.append(movie_data[feature])
            else:
                # Use median values for missing data
                if feature == 'budget':
                    num_data.append(15000000)  # Median budget
                elif feature == 'revenue':
                    num_data.append(25000000)  # Median revenue
                elif feature == 'runtime':
                    num_data.append(100)  # Median runtime
                elif feature == 'profit':
                    num_data.append(10000000)  # Median profit
                else:
                    num_data.append(0)
        
        num_encoded = scaler.transform([num_data])
        
        # Combine all features in the same order as training
        X = np.hstack([
            genres_encoded,
            keywords_encoded,
            cast_encoded,
            director_encoded,
            language_encoded,
            num_encoded
        ])
        
        return X
        
    except Exception as e:
        st.error(f"Error in preprocessing: {e}")
        return None
    
def predict_movie_rating(movies, models, movie_title):
    """Predict movie rating using Random Forest"""
    try:
        # Load preprocessing objects
        preprocessing_objects = load_preprocessing_objects()
        if not preprocessing_objects:
            st.error("Could not load preprocessing objects")
            return {}
        
        # Get movie data
        movie_data = movies[movies['title'] == movie_title].iloc[0].to_dict()
        
        # Preprocess the movie
        X = preprocess_movie_for_prediction(movie_data, preprocessing_objects)
        
        if X is None:
            return {}
        
        # Make prediction
        prediction_proba = models['random_forest'].predict_proba(X)[0]
        prediction_class = models['random_forest'].predict(X)[0]
        classes = models['random_forest'].classes_
        
        # Create result dictionary
        result = {
            'predicted_class': prediction_class,
            'probabilities': {}
        }
        
        for i, class_name in enumerate(classes):
            result['probabilities'][class_name] = prediction_proba[i]
        
        return result
        
    except Exception as e:
        st.error(f"Error in rating prediction: {e}")
        return {}

def show():
    st.header("Movie Recommendations")
    
    # Load data and models
    movies = load_data()
    models = load_models()
    
    if movies is None or not models:
        st.error("Could not load data or models. Please check file paths.")
        st.stop()
    
    st.write(f"Dataset contains {len(movies):,} movies")
    
    # Create tabs for different recommendation methods
    tab1, tab2, tab3 = st.tabs([
        "Content-Based (NLP)", 
        "Clustering-Based", 
        "Rating Prediction"
    ])
    
    with tab1:
        show_nlp_recommendations(movies, models)
    
    with tab2:
        show_clustering_recommendations(movies, models)
    
    with tab3:
        show_rating_prediction(movies, models)

def show_nlp_recommendations(movies, models):
    st.subheader("Content-Based Recommendations (NLP)")
    
    st.write("""
    This method uses TF-IDF vectorization and cosine similarity to find movies 
    with similar content based on genres, cast, keywords, and director.
    """)
    
    # Movie selection
    movie_titles = sorted(movies['title'].unique())
    selected_movie = st.selectbox(
        "Select a movie to get recommendations:",
        options=[""] + movie_titles,
        key="nlp_movie_select"
    )
    
    if selected_movie:
        # Display selected movie info
        movie_info = movies[movies['title'] == selected_movie].iloc[0]
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.write("**Selected Movie:**")
            st.write(f"**Title:** {movie_info['title']}")
            st.write(f"**Year:** {int(movie_info['release_year'])}")
            st.write(f"**Rating:** {movie_info['vote_average']}/10")
            st.write(f"**Director:** {movie_info['director']}")
        
        with col2:
            st.write(f"**Genres:** {', '.join(movie_info['genres'])}")
            if len(movie_info['top_cast']) > 0:
                st.write(f"**Cast:** {', '.join(movie_info['top_cast'][:3])}")
            if 'nlp' in movie_info:
                st.write(f"**NLP Features:** {movie_info['nlp'][:100]}...")
        
        # Get recommendations
        if st.button("Get NLP Recommendations", type="primary", key="nlp_recommend"):
            with st.spinner("Finding similar movies..."):
                recommendations = get_nlp_recommendations(movies, models, selected_movie)
                
                if not recommendations.empty:
                    st.subheader("🎬 Recommended Movies")
                    
                    for idx, movie in recommendations.iterrows():
                        with st.container():
                            col1, col2, col3 = st.columns([3, 1, 1])
                            
                            with col1:
                                st.write(f"**{movie['title']}** ({int(movie['release_year'])})")
                                st.write(f"*{', '.join(movie['genres'])}*")
                                st.write(f"Director: {movie['director']}")
                            
                            with col2:
                                st.metric("Rating", f"{movie['vote_average']}/10")
                                st.write(f"Similarity: {movie['similarity_score']:.3f}")
                            
                            st.divider()

def show_clustering_recommendations(movies, models):
    """Show clustering-based recommendations"""
    st.subheader("Clustering-Based Recommendations")
    
    st.write("""
    This method groups movies into clusters based on multiple features and recommends 
    movies from the same cluster as your selected movie.
    """)
    
    # Clustering method selection
    clustering_method = st.selectbox(
        "Select clustering method:",
        ["K-Means", "Mean Shift", "Hierarchical"],
        key="clustering_method"
    )
    
    # Movie selection
    movie_titles = sorted(movies['title'].unique())
    selected_movie = st.selectbox(
        "Select a movie to get recommendations:",
        options=[""] + movie_titles,
        key="cluster_movie_select"
    )
    
    if selected_movie:
        method_map = {
            "K-Means": "kmeans",
            "Mean Shift": "meanshift", 
            "Hierarchical": "hierarchical"
        }
        
        if st.button("Get Cluster Recommendations", type="primary", key="cluster_recommend"):
            with st.spinner(f"Finding movies in the same {clustering_method} cluster..."):
                recommendations = get_cluster_recommendations(
                    movies, models, selected_movie, method_map[clustering_method]
                )
                
                if not recommendations.empty:
                    st.subheader(f"🎬 Movies from the same {clustering_method} cluster")
                    
                    # Show cluster characteristics
                    st.write("**Cluster Characteristics:**")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        top_genres = pd.Series([g for genres in recommendations['genres'] for g in genres]).value_counts().head(3)
                        st.write("**Top Genres:**")
                        for genre, count in top_genres.items():
                            st.write(f"- {genre}: {count}")
                    
                    with col2:
                        top_directors = recommendations['director'].value_counts().head(3)
                        st.write("**Top Directors:**")
                        for director, count in top_directors.items():
                            st.write(f"- {director}: {count}")
                    
                    with col3:
                        avg_rating = recommendations['vote_average'].mean()
                        avg_year = recommendations['release_year'].mean()
                        st.write("**Cluster Stats:**")
                        st.write(f"- Avg Rating: {avg_rating:.1f}")
                        st.write(f"- Avg Year: {int(avg_year)}")
                    
                    st.divider()
                    
                    # Display recommendations
                    for idx, movie in recommendations.iterrows():
                        with st.container():
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                st.write(f"**{movie['title']}** ({int(movie['release_year'])})")
                                st.write(f"*{', '.join(movie['genres'])}*")
                                st.write(f"Director: {movie['director']}")
                            
                            with col2:
                                st.metric("Rating", f"{movie['vote_average']}/10")
                            
                            st.divider()

def show_rating_prediction(movies, models):
    """Show rating prediction interface"""
    st.subheader("Movie Rating Prediction")
    
    st.write("""
    Use our Random Forest model to predict how well a movie might be rated 
    based on its characteristics including genres, cast, director, and numerical features.
    """)
    
    # Movie selection for prediction
    movie_titles = sorted(movies['title'].unique())
    selected_movie = st.selectbox(
        "Select a movie to predict its rating:",
        options=[""] + movie_titles,
        key="rating_movie_select"
    )
    
    if selected_movie:
        movie_info = movies[movies['title'] == selected_movie].iloc[0]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Movie Information:**")
            st.write(f"**Title:** {movie_info['title']}")
            st.write(f"**Actual Rating:** {movie_info['vote_average']}/10")
            st.write(f"**Genres:** {', '.join(movie_info['genres'])}")
            st.write(f"**Director:** {movie_info['director']}")
            st.write(f"**Year:** {int(movie_info['release_year'])}")
            
            # Handle potential NaN values
            if pd.notna(movie_info['budget']) and movie_info['budget'] > 0:
                st.write(f"**Budget:** ${movie_info['budget']:,.0f}")
            else:
                st.write("**Budget:** Not available")
                
            if pd.notna(movie_info['revenue']) and movie_info['revenue'] > 0:
                st.write(f"**Revenue:** ${movie_info['revenue']:,.0f}")
            else:
                st.write("**Revenue:** Not available")
                
            if pd.notna(movie_info['runtime']) and movie_info['runtime'] > 0:
                st.write(f"**Runtime:** {movie_info['runtime']:.0f} minutes")
            else:
                st.write("**Runtime:** Not available")
        
        with col2:
            st.write("**Rating Categories:**")
            st.write("- **Good:** Rating > 7.0")
            st.write("- **Average:** Rating 5.0 - 7.0") 
            st.write("- **Poor:** Rating < 5.0")
            
            # Determine actual category
            actual_rating = movie_info['vote_average']
            if actual_rating > 7:
                actual_category = "Good"
            elif actual_rating >= 5:
                actual_category = "Average"
            else:
                actual_category = "Poor"
            
            st.write(f"**Actual Category:** {actual_category}")
            
            # Show top cast if available
            if len(movie_info['top_cast']) > 0:
                st.write(f"**Top Cast:** {', '.join(movie_info['top_cast'][:3])}")
            
            if st.button("Predict Rating Category", type="primary", key="predict_rating"):
                with st.spinner("Making prediction..."):
                    prediction_result = predict_movie_rating(movies, models, selected_movie)
                    
                    if prediction_result and 'predicted_class' in prediction_result:
                        st.subheader("Prediction Results")
                        
                        predicted_class = prediction_result['predicted_class']
                        probabilities = prediction_result['probabilities']
                        
                        # Show prediction
                        col3, col4 = st.columns(2)
                        
                        with col3:
                            st.metric("Predicted Category", predicted_class)
                            
                            # Check if prediction is correct
                            if predicted_class == actual_category:
                                st.success("✅ Correct prediction!")
                            else:
                                st.error("❌ Incorrect prediction")
                        
                        with col4:
                            st.write("**Prediction Confidence:**")
                            max_prob = max(probabilities.values())
                            confidence_color = "green" if max_prob > 0.6 else "orange" if max_prob > 0.4 else "red"
                            st.markdown(f"<h3 style='color: {confidence_color}'>{max_prob:.1%}</h3>", unsafe_allow_html=True)
                        
                        # Show all probabilities with progress bars
                        st.write("**Category Probabilities:**")
                        
                        # Sort probabilities for better display
                        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
                        
                        for category, prob in sorted_probs:
                            # Color code the progress bars
                            if category == actual_category:
                                st.markdown(f"**{category} (Actual):**")
                            else:
                                st.markdown(f"**{category}:**")
                            
                            st.progress(prob, text=f"{prob:.1%}")
                    
                    else:
                        st.error("Could not make prediction. Please try another movie or check if the model files are properly loaded.")
