"""
recommender.py
Music Recommendation System - ML Logic (Optimized for preprocessed data)
Updated with Remix Generation Feature
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
from fuzzywuzzy import process
import pickle
import warnings
warnings.filterwarnings('ignore')


class MusicRecommender:
    """Main class for music recommendation system"""
    
    def __init__(self, csv_path):
        """Initialize the recommender with preprocessed dataset path"""
        self.csv_path = csv_path
        self.df = None
        self.model = None
        self.scaler = None
        self.features = None
        self.remix_generator = None
        
    def load_preprocessed_data(self):
        """Load preprocessed dataset (much faster!)"""
        self.df = pd.read_csv(self.csv_path)
        
        # Load model, scaler, and features if they exist
        try:
            with open('model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            with open('scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            with open('features.pkl', 'rb') as f:
                self.features = pickle.load(f)
        except FileNotFoundError:
            # Model files don't exist yet, will train on demand
            pass
        
        return self.df
    
    def initialize_remix_generator(self):
        """Initialize the remix generator (lazy loading)"""
        if self.remix_generator is None:
            from remix_generator import RemixGenerator
            self.remix_generator = RemixGenerator(self.df)
            self.remix_generator.create_transactions()
            self.remix_generator.mine_associations()
        return self.remix_generator
    
    @staticmethod
    def clean_song_name(song_name):
        """Clean song names for matching"""
        if isinstance(song_name, str):
            return re.sub(r'[^a-z0-9]', '', song_name.lower())
        return None
    
    def find_song(self, song_name):
        """Find a song in the dataset with fuzzy matching"""
        cleaned_input = self.clean_song_name(song_name)
        if cleaned_input is None:
            return None, "invalid"
        
        self.df['cleaned_track_name'] = self.df['track_name'].apply(self.clean_song_name)
        song_row = self.df[self.df['cleaned_track_name'] == cleaned_input]
        
        match_type = "exact"
        
        if song_row.empty:
            fuzzy_match = process.extractOne(
                cleaned_input, 
                self.df['cleaned_track_name'], 
                scorer=process.fuzz.ratio
            )
            if fuzzy_match and fuzzy_match[1] >= 70:
                closest_match = fuzzy_match[0]
                song_row = self.df[self.df['cleaned_track_name'] == closest_match]
                match_type = "fuzzy"
            else:
                match_type = "not_found"
        
        self.df.drop(columns=['cleaned_track_name'], inplace=True)
        
        if not song_row.empty:
            return song_row.iloc[0], match_type
        return None, match_type
    
    def recommend_similar_songs(self, song_name, num_recommendations=10):
        """Recommend similar songs based on cosine similarity within same language"""
        song_data, match_type = self.find_song(song_name)
        
        if song_data is None:
            return pd.DataFrame(), match_type
        
        language = song_data['language']
        df_same_lang = self.df[self.df['language'] == language].copy()
        
        exclude_cols = ['year', 'key', 'mode', 'time_signature', 'fuzzy_mood_score']
        numerical_features = self.df.select_dtypes(include=np.number).columns.tolist()
        features_for_similarity = [col for col in numerical_features if col not in exclude_cols]
        
        song_features = song_data[features_for_similarity].values.reshape(1, -1)
        lang_features = df_same_lang[features_for_similarity]
        
        similarities = cosine_similarity(song_features, lang_features).flatten()
        df_same_lang['similarity_score'] = similarities
        
        # Exclude input song and sort
        recommendations = df_same_lang[
            df_same_lang['track_name'] != song_data['track_name']
        ].sort_values('similarity_score', ascending=False)
        
        # Remove duplicates
        recommendations.drop_duplicates(subset=['track_name'], keep='first', inplace=True)
        
        top_recommendations = recommendations.head(num_recommendations)[
            ['track_name', 'artist_name', 'album_name', 'language', 'similarity_score']
        ]
        
        return top_recommendations, match_type
    
    def recommend_by_artist(self, song_name, num_recommendations=10):
        """Recommend songs by the same artist using popularity prediction"""
        if self.model is None or self.scaler is None:
            # Fallback to simple similarity if model not loaded
            return self.recommend_similar_songs_by_artist(song_name, num_recommendations)
        
        song_data, match_type = self.find_song(song_name)
        
        if song_data is None:
            return pd.DataFrame(), match_type
        
        # Get all artists from the song
        artists = [a.strip().lower() for a in song_data['artist_name'].split(',')]
        
        # Find all songs featuring any of these artists
        df_artists = pd.DataFrame()
        for artist in artists:
            df_artists = pd.concat([
                df_artists, 
                self.df[self.df['artist_name'].str.lower().str.contains(artist, na=False)].copy()
            ])
        
        df_artists.drop_duplicates(subset=['track_id'], inplace=True)
        df_other = df_artists[df_artists['track_name'] != song_data['track_name']].copy()
        
        if df_other.empty:
            return pd.DataFrame(), match_type
        
        # Predict popularity
        X_other = self.scaler.transform(df_other[self.features])
        df_other['predicted_popularity'] = self.model.predict(X_other)
        
        # Calculate similarity
        song_features = self.scaler.transform(song_data[self.features].values.reshape(1, -1))
        df_other['similarity_score'] = cosine_similarity(song_features, X_other).flatten()
        
        # Combined ranking
        df_other['ranking_score'] = df_other['predicted_popularity'] * df_other['similarity_score']
        
        recommendations = df_other.sort_values('ranking_score', ascending=False)
        
        top_recommendations = recommendations.head(num_recommendations)[
            ['track_name', 'artist_name', 'album_name', 'predicted_popularity', 
             'similarity_score', 'ranking_score']
        ]
        
        return top_recommendations, match_type
    
    def recommend_similar_songs_by_artist(self, song_name, num_recommendations=10):
        """Fallback method: recommend similar songs by same artist without ML model"""
        song_data, match_type = self.find_song(song_name)
        
        if song_data is None:
            return pd.DataFrame(), match_type
        
        artists = [a.strip().lower() for a in song_data['artist_name'].split(',')]
        
        df_artists = pd.DataFrame()
        for artist in artists:
            df_artists = pd.concat([
                df_artists, 
                self.df[self.df['artist_name'].str.lower().str.contains(artist, na=False)].copy()
            ])
        
        df_artists.drop_duplicates(subset=['track_id'], inplace=True)
        df_other = df_artists[df_artists['track_name'] != song_data['track_name']].copy()
        
        if df_other.empty:
            return pd.DataFrame(), match_type
        
        # Sort by popularity
        df_other = df_other.sort_values('popularity', ascending=False)
        df_other['ranking_score'] = df_other['popularity']
        df_other['predicted_popularity'] = df_other['popularity']
        df_other['similarity_score'] = 0.8  # Default similarity
        
        top_recommendations = df_other.head(num_recommendations)[
            ['track_name', 'artist_name', 'album_name', 'predicted_popularity', 
             'similarity_score', 'ranking_score']
        ]
        
        return top_recommendations, match_type
    
    def generate_remix_playlists(self, song_name, num_remixes=7):
        """
        Generate remix playlists for a given song using Apriori algorithm
        
        Args:
            song_name: Name of the song to generate remixes for
            num_remixes: Number of remix variants (7-8)
            
        Returns:
            List of remix structures with enriched song data
        """
        generator = self.initialize_remix_generator()
        remixes = generator.generate_remixes_for_song(song_name, num_remixes)
        return remixes
    
    def get_top_songs_by_language(self, language, n=10):
        """Get top N songs for a specific language"""
        lang_df = self.df[self.df['language'] == language].copy()
        return lang_df.nlargest(n, 'popularity')[
            ['track_name', 'artist_name', 'album_name', 'popularity']
        ]
    
    def get_mood_playlist(self, mood, n=10):
        """Get songs for a specific mood"""
        if 'mood_label' not in self.df.columns:
            return pd.DataFrame()
        
        
        mood_df = self.df[self.df['mood_label'] == mood].copy()
        return mood_df.head(n)[['track_name', 'artist_name', 'language', 'mood_label']]
    
    def get_all_languages(self):
        """Get list of all languages in dataset"""
        return sorted(self.df['language'].unique().tolist())
    
    def get_all_moods(self):
        """Get list of all moods"""
        if 'mood_label' not in self.df.columns:
            return []
        return sorted(self.df['mood_label'].unique().tolist())
    
    def get_dataset_stats(self):
        """Get basic statistics about the dataset"""
        stats = {
            'total_songs': len(self.df),
            'total_artists': self.df['artist_name'].nunique(),
            'languages': self.df['language'].nunique(),
            'moods': self.df['mood_label'].nunique() if 'mood_label' in self.df.columns else 0
        }
        return stats


# Convenience function for direct use
def initialize_recommender(csv_path):
    """Initialize and prepare the recommender system (fast loading)"""
    recommender = MusicRecommender(csv_path)
    recommender.load_preprocessed_data()
    return recommender


if __name__ == "__main__":
    # Example usage
    csv_path = "spotify_tracks_processed.csv"
    
    print("Initializing Music Recommender System...")
    recommender = initialize_recommender(csv_path)
    
    print("\n" + "="*50)
    print("Dataset Statistics:")
    stats = recommender.get_dataset_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print("\n" + "="*50)
    print("Testing recommendation system...")
    
    # Test song recommendation
    test_song = input("\nEnter a song name to test: ")
    
    # Similar songs
    similar, match_type = recommender.recommend_similar_songs(test_song, 5)
    if not similar.empty:
        print(f"\nSimilar songs (Match type: {match_type}):")
        print(similar)
    
    # Artist recommendations
    artist_recs, _ = recommender.recommend_by_artist(test_song, 5)
    if not artist_recs.empty:
        print(f"\nMore from artist:")
        print(artist_recs)
    
    # Remix playlists
    print("\nGenerating remix playlists...")
    remixes = recommender.generate_remix_playlists(test_song, 7)
    if remixes:
        print(f"\nGenerated {len(remixes)} remix playlists!")
        for remix in remixes[:2]:  # Show first 2
            print(f"\nRemix {remix['remix_id']}:")
            for pos, track in remix['tracks'].items():
                print(f"  {pos}: {track['track_name']}")