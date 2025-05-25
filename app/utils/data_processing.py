import pandas as pd
import numpy as np
import ast
from typing import List, Dict, Any

def load_and_preprocess_data(file_path: str) -> pd.DataFrame:
    """
    Load and preprocess the movie dataset
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        Preprocessed DataFrame
    """
    try:
        movies = pd.read_csv(file_path)
        
        # Convert string columns back to lists
        list_columns = ['genres', 'top_cast', 'keywords']
        for col in list_columns:
            if col in movies.columns:
                movies[col] = movies[col].apply(
                    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
                )
        
        # Ensure proper data types
        if 'release_year' not in movies.columns and 'release_date' in movies.columns:
            movies['release_date'] = pd.to_datetime(movies['release_date'], errors='coerce')
            movies['release_year'] = movies['release_date'].dt.year
        
        # Calculate profit if not exists
        if 'profit' not in movies.columns and 'budget' in movies.columns and 'revenue' in movies.columns:
            movies['profit'] = movies['revenue'] - movies['budget']
        
        # Calculate decade if not exists
        if 'decade' not in movies.columns and 'release_year' in movies.columns:
            movies['decade'] = (movies['release_year'] // 10) * 10
            
        return movies
    except Exception as e:
        raise Exception(f"Error loading data: {e}")


def get_genre_statistics(movies: pd.DataFrame) -> Dict[str, Any]:
    genre_counts = {}
    genre_ratings = {}
    genre_budgets = {}
    
    for i, row in movies.iterrows():
        for genre in row['genres']:
            # Count
            genre_counts[genre] = genre_counts.get(genre, 0) + 1
            
            # Ratings
            if genre not in genre_ratings:
                genre_ratings[genre] = []
            genre_ratings[genre].append(row['vote_average'])
            
            # Budgets
            if row['budget'] > 0:
                if genre not in genre_budgets:
                    genre_budgets[genre] = []
                genre_budgets[genre].append(row['budget'])
    
    # Calculate averages
    genre_avg_ratings = {
        genre: np.mean(ratings) 
        for genre, ratings in genre_ratings.items() 
        if len(ratings) >= 10
    }
    
    genre_avg_budgets = {
        genre: np.mean(budgets) 
        for genre, budgets in genre_budgets.items() 
        if len(budgets) >= 10
    }
    
    return {
        'counts': genre_counts,
        'avg_ratings': genre_avg_ratings,
        'avg_budgets': genre_avg_budgets
    }

def get_temporal_statistics(movies: pd.DataFrame) -> Dict[str, pd.Series]:
    return {
        'movies_per_year': movies['release_year'].value_counts().sort_index(),
        'avg_rating_per_year': movies.groupby('release_year')['vote_average'].mean(),
        'avg_budget_per_year': movies.groupby('release_year')['budget'].mean(),
        'avg_revenue_per_year': movies.groupby('release_year')['revenue'].mean()
    }

def get_top_entities(movies: pd.DataFrame, entity_type: str, top_n: int = 20) -> pd.DataFrame:
    if entity_type == 'actors':
        entity_counts = {}
        for cast_list in movies['top_cast']:
            for actor in cast_list:
                entity_counts[actor] = entity_counts.get(actor, 0) + 1
                
    elif entity_type == 'directors':
        entity_counts = movies['director'].value_counts().to_dict()
        
    elif entity_type == 'keywords':
        entity_counts = {}
        for keywords_list in movies['keywords']:
            for keyword in keywords_list:
                entity_counts[keyword] = entity_counts.get(keyword, 0) + 1
    else:
        raise ValueError("entity_type must be 'actors', 'directors', or 'keywords'")
    
    entity_df = pd.DataFrame({
        entity_type.capitalize()[:-1]: list(entity_counts.keys()),
        'Count': list(entity_counts.values())
    }).sort_values('Count', ascending=False).head(top_n)
    
    return entity_df

def filter_movies(movies: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    filtered_movies = movies.copy()
    
    # Year range filter
    if 'year_range' in filters:
        min_year, max_year = filters['year_range']
        filtered_movies = filtered_movies[
            (filtered_movies['release_year'] >= min_year) & 
            (filtered_movies['release_year'] <= max_year)
        ]
    
    # Genre filter
    if 'genres' in filters and filters['genres']:
        filtered_movies = filtered_movies[
            filtered_movies['genres'].apply(
                lambda x: any(genre in x for genre in filters['genres'])
            )
        ]
    
    # Rating range filter
    if 'rating_range' in filters:
        min_rating, max_rating = filters['rating_range']
        filtered_movies = filtered_movies[
            (filtered_movies['vote_average'] >= min_rating) & 
            (filtered_movies['vote_average'] <= max_rating)
        ]
    
    # Budget range filter
    if 'budget_range' in filters:
        min_budget, max_budget = filters['budget_range']
        filtered_movies = filtered_movies[
            (filtered_movies['budget'] >= min_budget) & 
            (filtered_movies['budget'] <= max_budget)
        ]
    
    return filtered_movies
