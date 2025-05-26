import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
    
    # Data Quality Check
    st.subheader("Data Quality Overview")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'budget' in movies.columns:
            valid_budget = movies['budget'].notna() & (movies['budget'] > 0)
            budget_coverage = valid_budget.sum()
            st.metric("Movies with Budget Data", f"{budget_coverage:,}")
            st.write(f"({(budget_coverage/len(movies)*100):.1f}% coverage)")
    
    with col2:
        if 'revenue' in movies.columns:
            valid_revenue = movies['revenue'].notna() & (movies['revenue'] > 0)
            revenue_coverage = valid_revenue.sum()
            st.metric("Movies with Revenue Data", f"{revenue_coverage:,}")
            st.write(f"({(revenue_coverage/len(movies)*100):.1f}% coverage)")
    
    with col3:
        if 'budget' in movies.columns and 'revenue' in movies.columns:
            valid_financial = (movies['budget'].notna() & (movies['budget'] > 0) & 
                             movies['revenue'].notna() & (movies['revenue'] > 0))
            financial_coverage = valid_financial.sum()
            st.metric("Complete Financial Data", f"{financial_coverage:,}")
            st.write(f"({(financial_coverage/len(movies)*100):.1f}% coverage)")
    
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
            default=['title', 'release_year', 'genres', 'vote_average', 'budget', 'revenue', 'director']
        )
    
    if columns_to_show:
        st.dataframe(movies[columns_to_show].head(num_rows), use_container_width=True)
    else:
        st.dataframe(movies.head(num_rows), use_container_width=True)
    
    # Statistical summary
    st.subheader("Statistical Summary")
    
    numeric_cols = ['budget', 'revenue', 'runtime', 'vote_average', 'vote_count', 'profit', 'decade']
    available_numeric_cols = [col for col in numeric_cols if col in movies.columns]
    
    if available_numeric_cols:
        summary_stats = movies[available_numeric_cols].describe()
        st.dataframe(summary_stats, use_container_width=True)
    
    # Financial Analysis (without ROI)
    if 'budget' in movies.columns and 'revenue' in movies.columns:
        st.subheader("Financial Analysis")
        
        # Filter for movies with valid financial data
        financial_movies = movies[
            (movies['budget'].notna()) & (movies['budget'] > 0) &
            (movies['revenue'].notna()) & (movies['revenue'] > 0)
        ].copy()
        
        if len(financial_movies) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Budget Distribution**")
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Use log scale for better visualization
                budget_data = financial_movies['budget']
                ax.hist(np.log10(budget_data), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                ax.set_xlabel('Budget (Log10 Scale)')
                ax.set_ylabel('Number of Movies')
                ax.set_title('Distribution of Movie Budgets')
                
                # Add some statistics
                median_budget = budget_data.median()
                ax.axvline(np.log10(median_budget), color='red', linestyle='--', 
                          label=f'Median: ${median_budget:,.0f}')
                ax.legend()
                
                st.pyplot(fig)
                plt.close()
                
                # Budget statistics
                st.write("**Budget Statistics:**")
                st.write(f"- Median: ${budget_data.median():,.0f}")
                st.write(f"- Mean: ${budget_data.mean():,.0f}")
                st.write(f"- Min: ${budget_data.min():,.0f}")
                st.write(f"- Max: ${budget_data.max():,.0f}")
            
            with col2:
                st.write("**Revenue Distribution**")
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Use log scale for better visualization
                revenue_data = financial_movies['revenue']
                ax.hist(np.log10(revenue_data), bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
                ax.set_xlabel('Revenue (Log10 Scale)')
                ax.set_ylabel('Number of Movies')
                ax.set_title('Distribution of Movie Revenues')
                
                # Add some statistics
                median_revenue = revenue_data.median()
                ax.axvline(np.log10(median_revenue), color='red', linestyle='--', 
                          label=f'Median: ${median_revenue:,.0f}')
                ax.legend()
                
                st.pyplot(fig)
                plt.close()
                
                # Revenue statistics
                st.write("**Revenue Statistics:**")
                st.write(f"- Median: ${revenue_data.median():,.0f}")
                st.write(f"- Mean: ${revenue_data.mean():,.0f}")
                st.write(f"- Min: ${revenue_data.min():,.0f}")
                st.write(f"- Max: ${revenue_data.max():,.0f}")
            
            st.write("**Profitability Overview**")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                profitable_movies = len(financial_movies[financial_movies['revenue'] > financial_movies['budget']])
                profit_rate = (profitable_movies / len(financial_movies)) * 100
                st.metric("Profitable Movies", f"{profitable_movies:,}")
                st.write(f"({profit_rate:.1f}% of movies with financial data)")
            
            with col2:
                if 'profit' in financial_movies.columns:
                    avg_profit = financial_movies['profit'].mean()
                    st.metric("Average Profit", f"${avg_profit:,.0f}")
                else:
                    avg_profit = (financial_movies['revenue'] - financial_movies['budget']).mean()
                    st.metric("Average Profit", f"${avg_profit:,.0f}")
            
            with col3:
                top_grossing = financial_movies.loc[financial_movies['revenue'].idxmax()]
                st.metric("Highest Revenue", f"${financial_movies['revenue'].max():,.0f}")
                st.write(f"Movie: {top_grossing['title']}")
            
            # Top grossing movies
            st.write("**Top 10 Highest Grossing Movies:**")
            top_grossing_movies = financial_movies.nlargest(10, 'revenue')[['title', 'budget', 'revenue', 'release_year']]
            
            for idx, movie in top_grossing_movies.iterrows():
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**{movie['title']}** ({int(movie['release_year'])})")
                with col2:
                    st.write(f"Budget: ${movie['budget']/1e6:.1f}M")
                with col3:
                    st.write(f"Revenue: ${movie['revenue']/1e6:.1f}M")
            
        else:
            st.warning("No movies with complete financial data (budget > 0 and revenue > 0) found.")
    
    # Rating Analysis
    st.subheader("Rating Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Rating Distribution**")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(movies['vote_average'], bins=30, alpha=0.7, color='purple', edgecolor='black')
        ax.axvline(movies['vote_average'].mean(), color='red', linestyle='--', 
                  label=f'Mean: {movies["vote_average"].mean():.1f}')
        ax.axvline(movies['vote_average'].median(), color='blue', linestyle='--', 
                  label=f'Median: {movies["vote_average"].median():.1f}')
        ax.set_xlabel('Rating')
        ax.set_ylabel('Number of Movies')
        ax.set_title('Distribution of Movie Ratings')
        ax.legend()
        
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.write("**Rating Categories**")
        
        # Creating rating categories
        def categorize_rating(rating):
            if rating >= 7.0:
                return 'Good (7.0 and above)'
            elif rating >= 5.0:
                return 'Average (5.0-6.9)'
            else:
                return 'Poor (<5.0)'
        
        movies['rating_category'] = movies['vote_average'].apply(categorize_rating)
        rating_counts = movies['rating_category'].value_counts()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        rating_counts.plot(kind='bar', ax=ax, color='lightcoral')
        ax.set_title('Movies by Rating Category')
        ax.set_xlabel('Rating Category')
        ax.set_ylabel('Number of Movies')
        ax.tick_params(axis='x', rotation=45)
        
        st.pyplot(fig)
        plt.close()
        
        # Show percentages
        st.write("**Rating Breakdown:**")
        for category, count in rating_counts.items():
            percentage = (count / len(movies)) * 100
            st.write(f"- {category}: {count:,} ({percentage:.1f}%)")
    
    # Key Insights
    st.subheader("Key Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Financial Insights:**")
        if 'budget' in movies.columns and 'revenue' in movies.columns:
            financial_movies = movies[
                (movies['budget'].notna()) & (movies['budget'] > 0) &
                (movies['revenue'].notna()) & (movies['revenue'] > 0)
            ]
            
            if len(financial_movies) > 0:
                profitable_movies = len(financial_movies[financial_movies['revenue'] > financial_movies['budget']])
                profit_rate = (profitable_movies / len(financial_movies)) * 100
                st.write(f"- {profitable_movies:,} movies ({profit_rate:.1f}%) are profitable")
                
                # Average profit instead of ROI
                avg_profit = (financial_movies['revenue'] - financial_movies['budget']).mean()
                st.write(f"- Average profit: ${avg_profit:,.0f}")
                
                # Highest grossing movie
                top_grossing = financial_movies.loc[financial_movies['revenue'].idxmax()]
                st.write(f"- Highest grossing: {top_grossing['title']} (${top_grossing['revenue']:,.0f})")
                
                # Most profitable movie
                financial_movies['profit_calc'] = financial_movies['revenue'] - financial_movies['budget']
                most_profitable = financial_movies.loc[financial_movies['profit_calc'].idxmax()]
                st.write(f"- Most profitable: {most_profitable['title']} (${most_profitable['profit_calc']:,.0f} profit)")
            else:
                st.write("- No complete financial data available")
    
    with col2:
        st.write("**Rating Insights:**")
        if 'vote_average' in movies.columns:
            high_rated = len(movies[movies['vote_average'] >= 7.0])
            high_rated_rate = (high_rated / len(movies)) * 100
            st.write(f"- {high_rated:,} movies ({high_rated_rate:.1f}%) have rating ≥ 7.0")
            
            # Most common rating range
            rating_mode = movies['vote_average'].round(1).mode().iloc[0]
            st.write(f"- Most common rating: {rating_mode}")
            
            # Highest rated movie
            top_rated = movies.loc[movies['vote_average'].idxmax()]
            st.write(f"- Highest rated: {top_rated['title']} ({top_rated['vote_average']}/10)")
            
            # Average rating by decade
            if 'decade' in movies.columns:
                decade_ratings = movies.groupby('decade')['vote_average'].mean().sort_index()
                best_decade = decade_ratings.idxmax()
                st.write(f"- Best decade for ratings: {int(best_decade)}s ({decade_ratings[best_decade]:.1f} avg)")
    
    # NLP Features
    if 'nlp' in movies.columns:
        st.subheader("NLP Features")
        col1, col2 = st.columns(2)

        with col1:
            st.write("**NLP Column Info:**")
            st.write(f"- Contains processed text features")
            avg_length = movies['nlp'].str.len().mean()
            st.write(f"- Average length: {avg_length:.0f} characters")
            st.write(f"- Used for content-based recommendations")
            st.write("**Sample NLP Content:**")
            sample_nlp = movies['nlp'].iloc[0]
            if len(sample_nlp) > 100:
                sample_nlp = sample_nlp[:100] + "..."
            st.code(sample_nlp, language=None)
