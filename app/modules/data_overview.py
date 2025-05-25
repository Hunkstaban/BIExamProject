import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

from data_processing import load_and_preprocess_data

@st.cache_data
def load_data():
    """Load and cache the movie dataset"""
    try:
        return load_and_preprocess_data('../data/movies_with_nlp.csv')
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def show():
    """Display data overview page"""
    st.header("Dataset Overview")
    st.subheader("After cleaning, wrangling, and engineering the data it includes the following metrics")
    # Load data
    movies = load_data()
    if movies is None:
        st.stop()
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Movies", f"{len(movies):,}")
    with col2:
        unique_genres = len(set([genre for genres in movies['genres'] for genre in genres]))
        st.metric("Unique Genres", unique_genres)
    with col3:
        year_range = f"{int(movies['release_year'].min())}-{int(movies['release_year'].max())}"
        st.metric("Year Range", year_range)
    with col4:
        avg_rating = f"{movies['vote_average'].mean():.1f}"
        st.metric("Avg Rating", avg_rating)
    
    # Dataset information
    st.subheader("Dataset Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Dataset Shape:**")
        st.write(f"- Rows: {movies.shape[0]:,}")
        st.write(f"- Columns: {movies.shape[1]}")
        
        st.write("**Data Types:**")
        data_types = movies.dtypes.value_counts()
        for dtype, count in data_types.items():
            st.write(f"- {dtype}: {count} columns")
    
    with col2:
        st.write("**Missing Values:**")
        missing_values = movies.isnull().sum()
        missing_values = missing_values[missing_values > 0]
        if len(missing_values) > 0:
            for col, count in missing_values.items():
                percentage = (count / len(movies)) * 100
                st.write(f"- {col}: {count} ({percentage:.1f}%)")
        else:
            st.write("No missing values found!")
    
    # Sample data
    st.subheader("Sample Data")
    
    # Display options
    col1, col2 = st.columns(2)
    with col1:
        num_rows = st.slider("Number of rows to display", 5, 50, 10)
    with col2:
        columns_to_show = st.multiselect(
            "Select columns to display",
            options=movies.columns.tolist(),
            default=['title', 'release_year', 'genres', 'vote_average', 'budget', 'revenue', 'director']  # Updated defaults
        )

    
    if columns_to_show:
        st.dataframe(movies[columns_to_show].head(num_rows), use_container_width=True)
    else:
        st.dataframe(movies.head(num_rows), use_container_width=True)
    
    # Statistical summary
    st.subheader("Statistical Summary")

    # Updated to include new potential columns
    numeric_cols = ['budget', 'revenue', 'runtime', 'vote_average', 'vote_count', 'profit', 'decade']
    available_numeric_cols = [col for col in numeric_cols if col in movies.columns]

    if available_numeric_cols:
        summary_stats = movies[available_numeric_cols].describe()
        st.dataframe(summary_stats, use_container_width=True)
    
    if available_numeric_cols:
        summary_stats = movies[available_numeric_cols].describe()
        st.dataframe(summary_stats, use_container_width=True)
        
        # Additional insights
        st.subheader("Key Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Financial Insights:**")
            if 'budget' in movies.columns and 'revenue' in movies.columns:
                profitable_movies = len(movies[movies['profit'] > 0])
                profit_rate = (profitable_movies / len(movies)) * 100
                st.write(f"- {profitable_movies:,} movies ({profit_rate:.1f}%) are profitable")
                
                avg_roi = ((movies['revenue'] - movies['budget']) / movies['budget'] * 100).mean()
                st.write(f"- Average ROI: {avg_roi:.1f}%")
        
        with col2:
            st.write("**Rating Insights:**")
            if 'vote_average' in movies.columns:
                high_rated = len(movies[movies['vote_average'] >= 7.0])
                high_rated_rate = (high_rated / len(movies)) * 100
                st.write(f"- {high_rated:,} movies ({high_rated_rate:.1f}%) have rating ≥ 7.0")
                
                most_common_rating = movies['vote_average'].mode().iloc[0]
                st.write(f"- Most common rating: {most_common_rating}")
                
    # Add this after the existing insights section
    if 'nlp' in movies.columns:
        st.subheader("NLP Features")
        col1, col2 = st.columns(2)

        with col1:
            st.write("**NLP Column Info:**")
            st.write(f"- Contains processed text features")
            st.write(f"- Average length: {movies['nlp'].str.len().mean():.0f} characters")
    
        with col2:
            st.write("**Sample NLP Content:**")
            sample_nlp = movies['nlp'].iloc[0][:100] + "..." if len(movies['nlp'].iloc[0]) > 100 else movies['nlp'].iloc[0]
            st.write(f"- {sample_nlp}")

