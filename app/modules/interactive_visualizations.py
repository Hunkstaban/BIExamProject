import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

from data_processing import load_and_preprocess_data, get_genre_statistics, get_temporal_statistics, get_top_entities
from visualization import (create_temporal_chart, create_horizontal_bar_chart, 
                          create_scatter_plot, create_correlation_heatmap, 
                          create_distribution_plot, create_multi_line_chart)


@st.cache_data
def load_data():
    try:
        return load_and_preprocess_data('../data/movies_with_nlp.csv')
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def show():
    st.header("Interactive Data Visualizations")
    
    # Load data
    movies = load_data()
    if movies is None:
        st.stop()
    
    # Sidebar filters
    st.sidebar.subheader("Filters")
    
    # Year range filter
    min_year = int(movies['release_year'].min())
    max_year = int(movies['release_year'].max())
    year_range = st.sidebar.slider(
        "Select Year Range",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year)
    )
    
    # Genre filter
    all_genres = sorted(set([genre for genres in movies['genres'] for genre in genres]))
    selected_genres = st.sidebar.multiselect(
        "Select Genres",
        options=all_genres,
        default=[]
    )
    
    # Apply filters
    filtered_movies = movies[
        (movies['release_year'] >= year_range[0]) & 
        (movies['release_year'] <= year_range[1])
    ]
    
    if selected_genres:
        filtered_movies = filtered_movies[
            filtered_movies['genres'].apply(
                lambda x: any(genre in x for genre in selected_genres)
            )
        ]
    
    st.write(f"Showing {len(filtered_movies):,} movies (filtered from {len(movies):,} total)")
    
    # Create tabs for different visualization categories
    tab1, tab2, tab3, tab4 = st.tabs([
        "Temporal Analysis", 
        "Genre Analysis", 
        "Financial Analysis", 
        "Rating Analysis"
    ])
    
    with tab1:
        show_temporal_analysis(filtered_movies)
    
    with tab2:
        show_genre_analysis(filtered_movies)
    
    with tab3:
        show_financial_analysis(filtered_movies)
    
    with tab4:
        show_rating_analysis(filtered_movies)

def show_temporal_analysis(movies):
    st.subheader("Movies Over Time")
    
    # Get temporal statistics
    temporal_stats = get_temporal_statistics(movies)
    
    # Movies per year
    fig1 = create_temporal_chart(
        temporal_stats['movies_per_year'],
        "Number of Movies Released per Year",
        "Year",
        "Number of Movies"
    )
    st.plotly_chart(fig1, use_container_width=True)
    
    # Average ratings over time
    fig2 = create_temporal_chart(
        temporal_stats['avg_rating_per_year'],
        "Average Movie Ratings by Year",
        "Year",
        "Average Rating"
    )
    st.plotly_chart(fig2, use_container_width=True)
    
    # Budget and revenue evolution
    col1, col2 = st.columns(2)
    
    with col1:
        fig3 = create_temporal_chart(
            temporal_stats['avg_budget_per_year'],
            "Average Budget by Year",
            "Year",
            "Average Budget ($)"
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        fig4 = create_temporal_chart(
            temporal_stats['avg_revenue_per_year'],
            "Average Revenue by Year",
            "Year",
            "Average Revenue ($)"
        )
        st.plotly_chart(fig4, use_container_width=True)
    
    # Decade analysis
    st.subheader("Decade Analysis")
    movies['decade'] = (movies['release_year'] // 10) * 10
    decade_stats = movies.groupby('decade').agg({
        'title': 'count',
        'vote_average': 'mean',
        'budget': 'mean',
        'revenue': 'mean'
    }).round(2)
    decade_stats.columns = ['Movies Count', 'Avg Rating', 'Avg Budget', 'Avg Revenue']
    
    st.dataframe(decade_stats, use_container_width=True)

def show_genre_analysis(movies):
    st.subheader("Genre Analysis")
    
    # Get genre statistics
    genre_stats = get_genre_statistics(movies)
    
    # Genre popularity
    genre_df = pd.DataFrame({
        'Genre': list(genre_stats['counts'].keys()),
        'Count': list(genre_stats['counts'].values())
    }).sort_values('Count', ascending=False).head(20)
    
    fig1 = create_horizontal_bar_chart(
        genre_df, 'Count', 'Genre',
        "Top 20 Most Popular Genres",
        'viridis'
    )
    st.plotly_chart(fig1, use_container_width=True)
    
    # Average rating by genre
    if genre_stats['avg_ratings']:
        genre_rating_df = pd.DataFrame({
            'Genre': list(genre_stats['avg_ratings'].keys()),
            'Average_Rating': list(genre_stats['avg_ratings'].values())
        }).sort_values('Average_Rating', ascending=False)
        
        fig2 = create_horizontal_bar_chart(
            genre_rating_df, 'Average_Rating', 'Genre',
            "Average Rating by Genre",
            'RdYlGn'
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # Average budget by genre
    if genre_stats['avg_budgets']:
        genre_budget_df = pd.DataFrame({
            'Genre': list(genre_stats['avg_budgets'].keys()),
            'Average_Budget': list(genre_stats['avg_budgets'].values())
        }).sort_values('Average_Budget', ascending=False)
        
        fig3 = create_horizontal_bar_chart(
            genre_budget_df, 'Average_Budget', 'Genre',
            "Average Budget by Genre",
            'plasma'
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    # Genre combinations
    st.subheader("Genre Combinations")
    genre_combinations = {}
    for genres_list in movies['genres']:
        if len(genres_list) > 1:
            combo = ', '.join(sorted(genres_list))
            genre_combinations[combo] = genre_combinations.get(combo, 0) + 1
    
    if genre_combinations:
        combo_df = pd.DataFrame({
            'Combination': list(genre_combinations.keys()),
            'Count': list(genre_combinations.values())
        }).sort_values('Count', ascending=False).head(15)
        
        fig4 = create_horizontal_bar_chart(
            combo_df, 'Count', 'Combination',
            "Top 15 Genre Combinations",
            'magma'
        )
        st.plotly_chart(fig4, use_container_width=True)

def show_financial_analysis(movies):
    st.subheader("Financial Analysis")
    
    # Budget vs Revenue scatter plot
    valid_data = movies[(movies['budget'] > 0) & (movies['revenue'] > 0)]
    
    if len(valid_data) > 0:
        fig1 = create_scatter_plot(
            valid_data, 'budget', 'revenue',
            "Budget vs Revenue",
            hover_data=['title', 'vote_average'],
            log_x=True, log_y=True
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        # Most profitable movies
        top_profit = movies.nlargest(20, 'profit')
        
        fig2 = create_horizontal_bar_chart(
            top_profit, 'profit', 'title',
            "Top 20 Most Profitable Movies",
            'viridis'
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        # Distribution plots
        col1, col2 = st.columns(2)
        
        with col1:
            fig3 = create_distribution_plot(
                movies[movies['budget'] > 0]['budget'],
                "Budget Distribution",
                nbins=30, log_x=True
            )
            st.plotly_chart(fig3, use_container_width=True)
        
        with col2:
            fig4 = create_distribution_plot(
                movies[movies['revenue'] > 0]['revenue'],
                "Revenue Distribution",
                nbins=30, log_x=True
            )
            st.plotly_chart(fig4, use_container_width=True)
        
        # ROI Analysis
        st.subheader("Return on Investment (ROI) Analysis")
        movies_with_roi = movies[(movies['budget'] > 0) & (movies['revenue'] > 0)].copy()
        movies_with_roi['roi'] = ((movies_with_roi['revenue'] - movies_with_roi['budget']) / movies_with_roi['budget']) * 100
        
        # Top ROI movies
        top_roi = movies_with_roi.nlargest(20, 'roi')
        
        fig5 = create_horizontal_bar_chart(
            top_roi, 'roi', 'title',
            "Top 20 Movies by ROI (%)",
            'RdYlGn'
        )
        st.plotly_chart(fig5, use_container_width=True)

def show_rating_analysis(movies):
    """Show rating analysis visualizations"""
    st.subheader("Rating Analysis")
    
    # Rating distribution
    fig1 = create_distribution_plot(
        movies['vote_average'],
        "Distribution of Movie Ratings",
        nbins=20
    )
    st.plotly_chart(fig1, use_container_width=True)
    
    # Correlation heatmap
    numeric_cols = ['budget', 'revenue', 'runtime', 'vote_average', 'vote_count', 'release_year', 'profit']
    available_cols = [col for col in numeric_cols if col in movies.columns]
    
    if len(available_cols) > 1:
        corr_matrix = movies[available_cols].corr()
        fig2 = create_correlation_heatmap(corr_matrix, "Correlation Matrix of Numeric Variables")
        st.plotly_chart(fig2, use_container_width=True)
    
    # Top actors, directors, keywords
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Top Actors")
        top_actors = get_top_entities(movies, 'actors', 15)
        fig3 = create_horizontal_bar_chart(
            top_actors, 'Count', 'Actor',
            "Top 15 Most Frequent Actors",
            'plasma'
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        st.subheader("Top Directors")
        top_directors = get_top_entities(movies, 'directors', 15)
        fig4 = create_horizontal_bar_chart(
            top_directors, 'Count', 'Director',
            "Top 15 Most Frequent Directors",
            'viridis'
        )
        st.plotly_chart(fig4, use_container_width=True)
    
    with col3:
        st.subheader("Top Keywords")
        top_keywords = get_top_entities(movies, 'keywords', 15)
        fig5 = create_horizontal_bar_chart(
            top_keywords, 'Count', 'Keyword',
            "Top 15 Most Frequent Keywords",
            'magma'
        )
        st.plotly_chart(fig5, use_container_width=True)
    
    # High-rated movies analysis
    st.subheader("High-Rated Movies Analysis")
    high_rated = movies[movies['vote_average'] >= 7.5]
    
    if len(high_rated) > 0:
        st.write(f"Found {len(high_rated)} movies with rating ≥ 7.5")
        
        # Keywords in high-rated movies
        high_rated_keywords = [keyword for keywords_list in high_rated['keywords'] for keyword in keywords_list]
        if high_rated_keywords:
            keyword_counts = pd.Series(high_rated_keywords).value_counts().head(20)
            keyword_df = pd.DataFrame({
                'Keyword': keyword_counts.index,
                'Count': keyword_counts.values
            })
            
            fig6 = create_horizontal_bar_chart(
                keyword_df, 'Count', 'Keyword',
                "Top 20 Keywords in High-Rated Movies",
                'RdYlGn'
            )
            st.plotly_chart(fig6, use_container_width=True)
