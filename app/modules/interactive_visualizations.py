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
    
    # Filter for valid financial data
    valid_budget = movies[(movies['budget'].notna()) & (movies['budget'] > 0)]
    valid_revenue = movies[(movies['revenue'].notna()) & (movies['revenue'] > 0)]
    valid_both = movies[
        (movies['budget'].notna()) & (movies['budget'] > 0) & 
        (movies['revenue'].notna()) & (movies['revenue'] > 0)
    ]
    
    # Show data availability
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Movies with Budget Data", f"{len(valid_budget):,}")
    with col2:
        st.metric("Movies with Revenue Data", f"{len(valid_revenue):,}")
    with col3:
        st.metric("Movies with Both", f"{len(valid_both):,}")
    
    # Budget vs Revenue scatter plot
    if len(valid_both) > 0:
        fig1 = create_scatter_plot(
            valid_both, 'budget', 'revenue',
            "Budget vs Revenue",
            hover_data=['title', 'vote_average'],
            log_x=True, log_y=True
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        # Most profitable movies (only if we have profit data or can calculate it)
        if 'profit' in movies.columns:
            profitable_movies = movies[
                (movies['profit'].notna()) & (movies['profit'] > 0)
            ]
            if len(profitable_movies) > 0:
                top_profit = profitable_movies.nlargest(20, 'profit')
                
                fig2 = create_horizontal_bar_chart(
                    top_profit, 'profit', 'title',
                    "Top 20 Most Profitable Movies",
                    'viridis'
                )
                st.plotly_chart(fig2, use_container_width=True)
        else:
            # Calculate profit on the fly
            valid_both['calculated_profit'] = valid_both['revenue'] - valid_both['budget']
            profitable_movies = valid_both[valid_both['calculated_profit'] > 0]
            
            if len(profitable_movies) > 0:
                top_profit = profitable_movies.nlargest(20, 'calculated_profit')
                
                fig2 = create_horizontal_bar_chart(
                    top_profit, 'calculated_profit', 'title',
                    "Top 20 Most Profitable Movies",
                    'viridis'
                )
                st.plotly_chart(fig2, use_container_width=True)
        
        # Budget vs Revenue by decade 
        if len(valid_both) > 50:  
            st.subheader("Financial Trends by Decade")
            
            # Adding decade column
            valid_both_copy = valid_both.copy()
            valid_both_copy['decade'] = (valid_both_copy['release_year'] // 10) * 10
            
            # Group by decade
            decade_financial = valid_both_copy.groupby('decade').agg({
                'budget': ['mean', 'median', 'count'],
                'revenue': ['mean', 'median']
            }).round(0)
            
            # Flatten column names
            decade_financial.columns = ['Avg Budget', 'Median Budget', 'Movie Count', 'Avg Revenue', 'Median Revenue']
            
            st.dataframe(decade_financial, use_container_width=True)
            
            # Plot average budget and revenue by decade
            decade_avg = valid_both_copy.groupby('decade')[['budget', 'revenue']].mean().reset_index()
            
            fig5 = go.Figure()
            fig5.add_trace(go.Scatter(
                x=decade_avg['decade'],
                y=decade_avg['budget'],
                mode='lines+markers',
                name='Average Budget',
                line=dict(color='blue')
            ))
            fig5.add_trace(go.Scatter(
                x=decade_avg['decade'],
                y=decade_avg['revenue'],
                mode='lines+markers',
                name='Average Revenue',
                line=dict(color='green'),
                yaxis='y2'
            ))
            
            fig5.update_layout(
                title="Average Budget and Revenue by Decade",
                xaxis_title="Decade",
                yaxis=dict(title="Budget ($)", side="left"),
                yaxis2=dict(title="Revenue ($)", side="right", overlaying="y"),
                hovermode='x unified'
            )
            
            st.plotly_chart(fig5, use_container_width=True)
    
    else:
        st.warning("No movies with complete financial data (both budget and revenue > 0) found.")
        
        # Still show individual distributions if available
        col1, col2 = st.columns(2)
        
        with col1:
            if len(valid_budget) > 0:
                st.write(f"**Budget Distribution** ({len(valid_budget):,} movies)")
                fig3 = create_distribution_plot(
                    valid_budget['budget'],
                    "Budget Distribution (Log Scale)",
                    nbins=30, log_x=True
                )
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.warning("No valid budget data available")
        
        with col2:
            if len(valid_revenue) > 0:
                st.write(f"**Revenue Distribution** ({len(valid_revenue):,} movies)")
                fig4 = create_distribution_plot(
                    valid_revenue['revenue'],
                    "Revenue Distribution (Log Scale)",
                    nbins=30, log_x=True
                )
                st.plotly_chart(fig4, use_container_width=True)
            else:
                st.warning("No valid revenue data available")


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
