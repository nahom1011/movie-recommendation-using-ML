# 🎬 Advanced AI Movie & TV Recommender

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b)
![API](https://img.shields.io/badge/TMDB-API-green)

A "Senior-Level" Content-Based Recommendation System that goes beyond simple genre matching. This application uses **Natural Language Processing (NLP)** and **Cosine Similarity** to analyze the "DNA" of a movie or TV show—including its cast, director, keywords, and plot overview—to find semantically similar content.

## 🚀 Key Features

*   **🔍 Dynamic Search**: No longer limited to a static dataset. You can search for **any** movie or TV show (e.g., *Inception*, *The Walking Dead*, *Joker*).
*   **🧠 Advanced "Metadata Soup"**: The AI mimics a human understanding of content by analyzing:
    *   **Keywords**: Specific themes (e.g., "time travel", "apocalypse", "mental illness").
    *   **Cast**: Top actors (e.g., "Keanu Reeves", "Leonardo DiCaprio").
    *   **Director/Creator**: The visionary behind the work.
    *   **Genres**: Broad categories.
    *   **Overview**: Plot description.
*   **⚡ Real-Time Indexing**: If you search for a title that isn't in the database, the system fetches it live from the TMDB API, builds its profile, and computes matches on the fly.
*   **🎨 Modern UI**: Features a clean, dual-section interface with detailed content profiles and poster-based recommendation grids.

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd movie-recommendation-using-ML
```

### 2. Install Dependencies
Make sure you have Python installed. Then run:
```bash
pip install -r requirements.txt
```

### 3. API Configuration (Crucial!)
You need a **TMDB API Key** to fetch data.
1.  Get your key from [The Movie Database (TMDB)](https://www.themoviedb.org/documentation/api).
2.  Create a file named `.env` in the root directory.
3.  Add your key inside:
    ```env
    TMDB_API_KEY=your_actual_api_key_here
    ```

### 4. Run the App
```bash
streamlit run app.py
```

## 🧠 How It Works

1.  **Data Fetching**: The `TMDBClient` fetches metadata (Cast, Crew, Keywords) from the API.
2.  **Soup Creation**: The `RecommendationEngine` combines all text features into a single string (the "soup").
    *   *Example Soup for 'Inception'*: "adventure sciencefiction action christophernolan leonardodicaprio josephgordon-levitt elliotpage dream subconscious thief..."
3.  **Vectorization**: `CountVectorizer` converts this text soup into a numerical matrix.
4.  **Similarity**: `Cosine Similarity` calculates the angle between vectors to determine how closely related two titles are.

## � Project Structure

*   `app.py`: Main Streamlit application and UI logic.
*   `recommendation_system.py`: The core AI engine handling data processing and similarity math.
*   `requirements.txt`: List of python dependencies.
*   `.env`: (Ignored by Git) Stores your API credentials securely.

---
*Powered by TMDB API.*