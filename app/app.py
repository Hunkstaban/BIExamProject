import streamlit as st
import sys
import os

# Add the utils folder to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

# Page configuration
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .welcome-intro {
        background-color: #e8f4fd;
        padding: 2rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin-bottom: 2rem;
        text-align: center;
        color: #333333;
    }
    .welcome-intro h2 {
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .welcome-intro p {
        color: #333333;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
     .recommendation-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">BI Exam - Movie Recommendation System</h1>', 
                unsafe_allow_html=True)
    
    # Initialize session state for page navigation
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Welcome"
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    st.sidebar.markdown("---")
    
    # Navigation buttons
    pages = [
        "Data Overview",
        "Interactive Visualizations",
        "Movie Recommendations"
    ]
    
    # Create navigation buttons
    for page in pages:
        if st.sidebar.button(page, key=f"nav_{page}", use_container_width=True):
            st.session_state.current_page = page
    
    # Show welcome message only on initial load
    if st.session_state.current_page == "Welcome":
        st.markdown("""
        <div class="welcome-intro">
            <h2>Welcome! 👋</h2>
            <p>This is my exam project for <strong>Business Intelligence</strong> for the 4th Semester of Datamatiker in CphBusiness.</p>
            <p>Use the navigation buttons on the left to explore my solution.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Import and display the selected page
    elif st.session_state.current_page == "Data Overview":
        from modules.data_overview import show
        show()
    elif st.session_state.current_page == "Interactive Visualizations":
        from modules.interactive_visualizations import show
        show()
    elif st.session_state.current_page == "Movie Recommendations":
        from modules.recommendations import show
        show()

if __name__ == "__main__":
    main()
