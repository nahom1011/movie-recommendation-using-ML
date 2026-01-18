import pandas as pd
import numpy as np
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

class TMDBClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.themoviedb.org/3"
        self.image_base_url = "https://image.tmdb.org/t/p/w500"

    def fetch_data(self, endpoint, pages=3):
        """Fetches multiple pages of data."""
        results = []
        for page in range(1, pages + 1):
            try:
                url = f"{self.base_url}{endpoint}?api_key={self.api_key}&language=en-US&page={page}"
                response = requests.get(url)
                response.raise_for_status()
                data = response.json()
                results.extend(data.get('results', []))
            except Exception as e:
                print(f"Error fetching {endpoint} page {page}: {e}")
        return results

    def search_multi(self, query):
        """Searches for movies and TV shows."""
        try:
            url = f"{self.base_url}/search/multi?api_key={self.api_key}&language=en-US&query={query}&page=1&include_adult=false"
            response = requests.get(url)
            response.raise_for_status()
            return response.json().get('results', [])
        except:
            return []

    def get_details(self, media_id, media_type):
        """Fetches detailed info including cast, crew, and keywords."""
        try:
            url = f"{self.base_url}/{media_type}/{media_id}?api_key={self.api_key}&append_to_response=credits,keywords"
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except:
            return None

class RecommendationEngine:
    def __init__(self, api_key):
        self.client = TMDBClient(api_key)
        self.df = pd.DataFrame()
        self.cosine_sim = None
        self.indices = None
        self.genre_map = {}
        
        # Initialize
        self._load_base_corpus()

    def _load_base_corpus(self):
        """Loads a base corpus of Popular/Top Rated items to have a recommendation pool."""
        print("Fetching base corpus...")
        movies = self.client.fetch_data("/movie/popular", pages=2)
        top_movies = self.client.fetch_data("/movie/top_rated", pages=1)
        tv = self.client.fetch_data("/tv/popular", pages=2)
        top_tv = self.client.fetch_data("/tv/top_rated", pages=1)
        
        raw_data = movies + top_movies + tv + top_tv
        # Deduplicate by ID
        unique_data = {item['id']: item for item in raw_data}.values()
        
        # Process into standard format
        processed_data = []
        for item in unique_data:
            processed = self._process_tmdb_item(item)
            if processed:
                processed_data.append(processed)
                
        self.df = pd.DataFrame(processed_data)
        # Drop duplicates based on ID just in case
        self.df.drop_duplicates(subset=['id', 'type'], inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        
        self._update_soup_and_sim()
        print(f"Engine initialized with {len(self.df)} titles.")

    def _process_tmdb_item(self, item, details=None):
        """Normalizes API data into our DataFrame schema."""
        try:
            # Determine type
            media_type = item.get('media_type')
            if not media_type:
                media_type = 'movie' if 'title' in item else 'tv'
            
            if media_type not in ['movie', 'tv']:
                return None

            title = item.get('title') if media_type == 'movie' else item.get('name')
            release_date = item.get('release_date') if media_type == 'movie' else item.get('first_air_date')
            
            # Extract basic genres if details not provided
            genres = []
            if 'genres' in item: # Detail view structure
                genres = [g['name'] for g in item['genres']]
            elif 'genre_ids' in item: # List view structure
                 # We need a map for this, but for now we'll skip if map missing or fetch map.
                 # Optimization: For base corpus, we might lack names.
                 # Let's rely on 'details' for the TARGET item, and for corpus items we do our best.
                 pass

            # Extract Soup Components
            overview = item.get('overview', '')
            keywords = []
            cast = []
            director = []
            
            if details:
                # Extract rich metadata
                if 'keywords' in details:
                    k_list = details['keywords'].get('keywords', []) if media_type == 'movie' else details['keywords'].get('results', [])
                    keywords = [k['name'] for k in k_list]
                
                if 'credits' in details:
                    # Cast (Top 3)
                    cast = [c['name'] for c in details['credits'].get('cast', [])[:3]]
                    # Director / Creator
                    if media_type == 'movie':
                        director = [c['name'] for c in details['credits'].get('crew', []) if c['job'] == 'Director']
                    else:
                        director = [c['name'] for c in details['credits'].get('crew', []) if c['job'] == 'Executive Producer'] # TV equivalent-ish
                        # Also created_by
                        if 'created_by' in details:
                            director.extend([c['name'] for c in details['created_by']])
            
            return {
                'id': item['id'],
                'tmdb_id': item['id'], # Distinct
                'title': title,
                'type': 'Movie' if media_type == 'movie' else 'TV Show',
                'media_type': media_type,
                'overview': overview,
                'vote_average': item.get('vote_average', 0),
                'vote_count': item.get('vote_count', 0),
                'poster_path': self.client.image_base_url + item['poster_path'] if item.get('poster_path') else None,
                'release_year': release_date.split('-')[0] if release_date else 'N/A',
                'keywords': keywords,
                'cast': cast,
                'director': director,
                'genres': genres,
                'soup_components': {
                    'overview': overview,
                    'keywords': keywords,
                    'cast': cast,
                    'director': director,
                    'genres': genres
                }
            }
        except Exception as e:
            # print(f"Error processing item: {e}")
            return None

    def _update_soup_and_sim(self):
        """Constructs the metadata soup and computes similarity matrix."""
        def create_soup(x):
            components = x['soup_components']
            # We join everything into one lowercased string
            # Names are stripped of spaces to treat "Johnny Depp" as "johnnydepp" (unique token)
            
            keywords_str = ' '.join([k.lower().replace(" ", "") for k in components['keywords']])
            cast_str = ' '.join([c.lower().replace(" ", "") for c in components['cast']])
            director_str = ' '.join([d.lower().replace(" ", "") for d in components['director']])
            genres_str = ' '.join([g.lower().replace(" ", "") for g in components.get('genres', [])]) # Handle missing safely
            overview_str = components['overview'].lower() if components['overview'] else ""
            
            # Simple list-based genres if genre_ids used (fallback)
            # For the base corpus items which lack 'details', keys might be empty.
            # This is a trade-off. The TARGET item will have rich soup.
            # The corpus items need rich soup to match well.
            # Ideally, we fetch details for ALL corpus items. But that's 600 calls.
            # Workaround: For corpus items, we rely heavily on Overview.
            
            return f"{keywords_str} {cast_str} {director_str} {genres_str} {overview_str}"

        self.df['soup'] = self.df.apply(create_soup, axis=1)
        
        # Vectorize
        count = CountVectorizer(stop_words='english')
        count_matrix = count.fit_transform(self.df['soup'])
        
        # Compute Sim
        self.cosine_sim = cosine_similarity(count_matrix, count_matrix)
        
        # Update indices
        self.df = self.df.reset_index(drop=True)
        self.indices = pd.Series(self.df.index, index=self.df['tmdb_id'])

    def search_titles(self, query):
        """Searches API and returns candidates."""
        results = self.client.search_multi(query)
        candidates = []
        for res in results:
            if res.get('media_type') not in ['movie', 'tv']: continue
            candidates.append({
                'id': res['id'],
                'title': res.get('title') if res.get('media_type') == 'movie' else res.get('name'),
                'type': res['media_type'],
                'year': (res.get('release_date') if res.get('media_type') == 'movie' else res.get('first_air_date') or "")[:4]
            })
        return candidates

    def select_title(self, tmdb_id, media_type):
        """Fetches full details for a title, adds to corpus, and returns it."""
        # Check if already exists AND has rich details (we assume items added via this method have details)
        # Actually, base corpus items dont have rich details. 
        # So we should always upgrade the item if we select it.
        
        full_details = self.client.get_details(tmdb_id, media_type)
        if not full_details: return None
        
        # Process
        item_data = self._process_tmdb_item(full_details, details=full_details)
        if not item_data: return None
        
        # Check existence by ID and Type (to handle ID collisions between Movie/TV although unlikely in TMDB unique IDs? No, IDs can collide across types)
        # TMDB IDs are unique per type.
        
        mask = (self.df['tmdb_id'] == tmdb_id) & (self.df['media_type'] == media_type)
        if mask.any():
            # Update existing
            idx = self.df[mask].index[0]
            # Update columns
            for k, v in item_data.items():
                self.df.at[idx, k] = v
        else:
            # Append new
            self.df = pd.concat([self.df, pd.DataFrame([item_data])], ignore_index=True)
            
        # Re-compute soup/sim
        # Optimization: We could just compute vector for this one item, but full re-compute is safer for Prototype
        self._update_soup_and_sim()
        
        return item_data

    def get_recommendations(self, tmdb_id, media_type, content_type_filter="Both"):
        """Get recommendations for a specific item in the corpus."""
        # Find index
        mask = (self.df['tmdb_id'] == tmdb_id) & (self.df['media_type'] == media_type)
        if not mask.any(): return pd.DataFrame()
        
        idx = self.df[mask].index[0]
        
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get top 20 candidates (excluding self)
        sim_scores = [s for s in sim_scores if s[0] != idx][:50]
        
        movie_indices = [i[0] for i in sim_scores]
        candidates = self.df.iloc[movie_indices].copy()
        
        # Filter
        if content_type_filter == "Movie":
            candidates = candidates[candidates['media_type'] == 'movie']
        elif content_type_filter == "TV Show":
            candidates = candidates[candidates['media_type'] == 'tv']
            
        return candidates.head(10)
