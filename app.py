import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from recommendation_system import RecommendationEngine

# Load environment variables
load_dotenv()

# Page Config
st.set_page_config(page_title="Advanced AI Recommender", page_icon="🧠", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .big-font { font-size:30px !important; font-weight: bold; color: #01b4e4; }
    .header-text { font-size:24px; font-weight: bold; margin-bottom: 10px; }
    .sub-text { color: #ccc; }
    .stButton>button { width: 100%; border-radius: 5px; }
    .metric-card { background-color: #222; padding: 10px; border-radius: 8px; text-align: center; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_engine():
    api_key = os.getenv("TMDB_API_KEY")
    if not api_key:
        api_key = "82d513dd1e80b7bbfc3bdfd8b60e90d0" # Fallback for demo if env missing
    return RecommendationEngine(api_key)

try:
    with st.spinner("Initializing AI Engine & Fetching Base Corpus..."):
        engine = load_engine()
except Exception as e:
    st.error(f"Failed to load engine: {e}")
    st.stop()

# Sidebar
st.sidebar.title("AI Settings")
st.sidebar.info("This system uses a **Content-Based Filtering** engine that analyzes plot, keywords, cast, and director to find similarities.")
content_type_filter = st.sidebar.radio("Recommendation Type", ["Both", "Movie", "TV Show"])

# Main Layout
st.markdown('<p class="big-font">Advanced Movie & TV Recommender</p>', unsafe_allow_html=True)

# 1. Search Interface
query = st.text_input("🔍 Search for a Title (Movie or TV Show)", placeholder="e.g., Inception, The Walking Dead, Breaking Bad...")

if query:
    # Fetch candidates
    candidates = engine.search_titles(query)
    
    if not candidates:
        st.warning("No titles found. Try a different query.")
    else:
        # Use a selectbox to let user disambiguate
        # Format: "Title (Year) - Type"
        options = {f"{c['title']} ({c['year']}) - {c['type'].upper()}": c for c in candidates}
        selected_label = st.selectbox("Did you mean:", list(options.keys()))
        
        if selected_label:
            selection = options[selected_label]
            
            # Fetch full details & Compute Sim
            with st.spinner(f"Analyzing '{selection['title']}' content profile..."):
                details = engine.select_title(selection['id'], selection['type'])
            
            if details:
                # --- SECTION 1: SELECTED TITLE INFO ---
                st.markdown("---")
                st.markdown('<p class="header-text">🎬 About this Title</p>', unsafe_allow_html=True)
                
                c1, c2 = st.columns([1, 2.5])
                with c1:
                    if details['poster_path']:
                        st.image(details['poster_path'], width=300)
                with c2:
                    st.markdown(f"## {details['title']}")
                    st.markdown(f"**{details['type']}** • {details['release_year']} • ⭐ {details['vote_average']}/10")
                    st.markdown(f"**Genres:** {', '.join(details['genres'])}")
                    st.markdown(f"_{details['overview']}_")
                    
                    # Show Soup Components for "Transparency"
                    with st.expander("See AI Content Profile (Debug)"):
                        st.write(f"**Keywords**: {', '.join(details['keywords'])}")
                        st.write(f"**Cast**: {', '.join(details['cast'])}")
                        st.write(f"**Director/Creator**: {', '.join(details['director'])}")

                # --- SECTION 2: RECOMMENDATIONS ---
                st.markdown("---")
                st.markdown('<p class="header-text">🎯 Similar Titles You May Like</p>', unsafe_allow_html=True)
                
                # Get Recs
                recs = engine.get_recommendations(selection['id'], selection['type'], content_type_filter)
                
                if recs.empty:
                    st.info("No similar titles found in the current corpus.")
                else:
                    cols = st.columns(5)
                    for i, (idx, row) in enumerate(recs.iterrows()):
                        col = cols[i % 5]
                        with col:
                            poster = row['poster_path'] if row['poster_path'] else "https://via.placeholder.com/500x750?text=No+Poster"
                            st.image(poster, use_column_width=True) # use_column_width auto-stretches which is fine, or just default
                            st.markdown(f"**{row['title']}**")
                            st.caption(f"{row['type']} • {row['release_year']}")
            else:
                st.error("Failed to fetch details for this title.")
else:
    st.info("👋 Type a movie or TV show title above to start.")
    
    # Optional: Show popular items as default
    st.markdown("---")
    st.markdown("**Example searches:**")
    st.markdown("- *Inception* (Sci-Fi/Action)")
    st.markdown("- *The Office* (Comedy TV)")
    st.markdown("- *The Notebook* (Romance)")
