# TMDB Movie Recommendation System

This application is a smart movie and TV show recommender engine powered by the **TMDB (The Movie Database) API**. It uses Machine Learning to suggest titles similar to the ones you like.

## 🚀 Key Features

*   **Live Data**: Fetches the latest "Popular" and "Top Rated" movies and TV shows directly from TMDB. No outdated CSV files!
*   **Smart Recommendations**: Uses **Content-Based Filtering** (Cosine Similarity) to find similar titles based on their **Overview** (plot) and **Genres**.
*   **Weighted Ratings**: Ranks recommendations using the IMDb weighted rating formula to ensure you see high-quality content, not just obscure matches.
*   **Rich UI**:
    *   some beautiful **Poster Grids** for browsing.
    *   **Detailed View** with Release Year, Rating, and Plot Summary.
    *   **Search** functionality to find specific titles in the loaded database.
*   **Discovery**: a "Discover by Genre" mode to find the best-rated content in specific categories (e.g., Action, Comedy).

## 🛠️ How It Works

1.  **Data Ingestion**: On startup, the app calls the TMDB API to download a fresh batch of movies and TV shows.
2.  **Processing**: It creates a "metadata soup" for each title (combining the plot description and genres).
3.  **Machine Learning**: It uses `CountVectorizer` to convert this text into numbers and calculates the **Cosine Similarity** between every pair of titles.
4.  **Ranking**: When you select a movie, it looks up the most similar titles, filters them (e.g., by type), and then ranks them by their Weighted Rating.

## 📦 How to Run

1.  **Install Requirements**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run the App**:
    ```bash
    streamlit run app.py
    ```

## 🔑 Configuration

The application uses a TMDB API Key to fetch data.
*   **Default Key**: A valid key is currently hardcoded in `app.py` for demonstration.
*   **Custom Key**: You can modify `load_engine()` in `app.py` to use your own key.