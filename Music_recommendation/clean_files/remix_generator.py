"""
remix_generator.py
Remix Playlist Generator using Apriori Algorithm for Association Rule Mining
"""

import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import warnings
warnings.filterwarnings('ignore')


class RemixGenerator:
    """Generate remix playlists using Apriori algorithm"""
    
    def __init__(self, df):
        """
        Initialize the remix generator
        
        Args:
            df: DataFrame with song data
        """
        self.df = df
        self.transactions = None
        self.frequent_itemsets = None
        self.rules = None
        
    def create_transactions(self):
        """
        Create transaction data from songs
        Groups songs by artist, album, and similar audio features to simulate playlists
        """
        transactions = []
        
        # Method 1: Group by artist (songs often played together)
        artist_groups = self.df.groupby('artist_name')['track_name'].apply(list)
        for songs in artist_groups:
            if len(songs) >= 2:  # Only keep groups with 2+ songs
                transactions.append(songs[:10])  # Limit to 10 songs per transaction
        
        # Method 2: Group by similar audio features (energy, danceability, valence)
        # Create bins for similarity grouping
        if all(col in self.df.columns for col in ['energy', 'danceability', 'valence']):
            self.df['energy_bin'] = pd.qcut(self.df['energy'], q=5, labels=False, duplicates='drop')
            self.df['dance_bin'] = pd.qcut(self.df['danceability'], q=5, labels=False, duplicates='drop')
            self.df['valence_bin'] = pd.qcut(self.df['valence'], q=5, labels=False, duplicates='drop')
            
            # Group by similar characteristics
            feature_groups = self.df.groupby(['energy_bin', 'dance_bin', 'valence_bin'])['track_name'].apply(list)
            for songs in feature_groups:
                if len(songs) >= 3:
                    transactions.append(songs[:8])
        
        # Method 3: Group by language and mood
        if 'language' in self.df.columns and 'mood_label' in self.df.columns:
            mood_groups = self.df.groupby(['language', 'mood_label'])['track_name'].apply(list)
            for songs in mood_groups:
                if len(songs) >= 3:
                    transactions.append(songs[:8])
        
        # Method 4: Random playlists (simulate user listening patterns)
        num_random_playlists = min(500, len(self.df) // 10)
        for _ in range(num_random_playlists):
            playlist_size = np.random.randint(3, 8)
            random_songs = self.df.sample(playlist_size)['track_name'].tolist()
            transactions.append(random_songs)
        
        self.transactions = transactions
        return transactions
    
    def mine_associations(self, min_support=0.01, min_confidence=0.3, min_lift=1.2):
        """
        Mine association rules using Apriori algorithm
        
        Args:
            min_support: Minimum support threshold (default: 0.01 = 1%)
            min_confidence: Minimum confidence threshold (default: 0.3 = 30%)
            min_lift: Minimum lift threshold (default: 1.2)
        """
        if self.transactions is None:
            self.create_transactions()
        
        # Convert transactions to one-hot encoded format
        te = TransactionEncoder()
        te_ary = te.fit(self.transactions).transform(self.transactions)
        df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
        
        # Apply Apriori algorithm
        self.frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
        
        if len(self.frequent_itemsets) == 0:
            print(f"No frequent itemsets found with min_support={min_support}")
            return None
        
        # Generate association rules
        self.rules = association_rules(
            self.frequent_itemsets, 
            metric="confidence", 
            min_threshold=min_confidence
        )
        
        # Filter by lift
        self.rules = self.rules[self.rules['lift'] >= min_lift]
        
        # Sort by confidence and lift
        self.rules = self.rules.sort_values(['confidence', 'lift'], ascending=False)
        
        return self.rules
    
    def get_associated_songs(self, song_name, top_n=12):
        """
        Get songs strongly associated with the given song
        
        Args:
            song_name: Name of the song to find associations for
            top_n: Number of associated songs to return
            
        Returns:
            List of associated song names with their scores
        """
        if self.rules is None or len(self.rules) == 0:
            # Try with lower thresholds if no rules found
            self.mine_associations(min_support=0.005, min_confidence=0.2, min_lift=1.1)
        
        if self.rules is None or len(self.rules) == 0:
            # If still no rules, return random songs from same artist/language
            song_data = self.df[self.df['track_name'] == song_name]
            if not song_data.empty:
                song_info = song_data.iloc[0]
                same_lang = self.df[
                    (self.df['language'] == song_info['language']) & 
                    (self.df['track_name'] != song_name)
                ]
                if not same_lang.empty:
                    samples = same_lang.sample(min(top_n, len(same_lang)))
                    return [(row['track_name'], 0.5) for _, row in samples.iterrows()]
            return []
        
        associated_songs = []
        
        # Find rules where the song is in the antecedent
        for _, rule in self.rules.iterrows():
            antecedents = rule['antecedents']
            consequents = rule['consequents']
            
            if song_name in antecedents:
                # Add consequent songs
                for consequent in consequents:
                    if consequent != song_name:
                        score = (rule['confidence'] * 0.6 + rule['lift'] * 0.4) / 2
                        associated_songs.append((consequent, score))
            
            elif song_name in consequents:
                # Also consider reverse associations
                for antecedent in antecedents:
                    if antecedent != song_name:
                        score = (rule['confidence'] * 0.5 + rule['lift'] * 0.3) / 2
                        associated_songs.append((antecedent, score))
        
        # Remove duplicates and sort by score
        unique_songs = {}
        for song, score in associated_songs:
            if song not in unique_songs or unique_songs[song] < score:
                unique_songs[song] = score
        
        sorted_songs = sorted(unique_songs.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_songs[:top_n]
    
    def create_remix_structure(self, associated_songs, num_remixes=7):
        """
        Create remix structures with Intro, Build-Up, Main Drop, and Outro
        
        Args:
            associated_songs: List of (song_name, score) tuples
            num_remixes: Number of remix variants to generate
            
        Returns:
            List of remix structures
        """
        if len(associated_songs) < 3:
            return []
        
        remixes = []
        
        for i in range(num_remixes):
            # Each remix has 3-4 songs
            remix_size = np.random.choice([3, 4], p=[0.4, 0.6])
            
            if len(associated_songs) < remix_size:
                break
            
            # Select songs for this remix
            # Use different strategies for variety
            if i == 0:
                # First remix: Top scoring songs
                selected = associated_songs[:remix_size]
            elif i == 1:
                # Second remix: Mix high and medium scores
                high = associated_songs[:2]
                medium = associated_songs[2:remix_size]
                selected = high + medium
            else:
                # Random selection with weighted probability
                songs_for_selection = associated_songs[:min(len(associated_songs), 12)]
                scores = np.array([score for _, score in songs_for_selection])
                if scores.sum() > 0:
                    probabilities = scores / scores.sum()
                else:
                    probabilities = np.ones(len(scores)) / len(scores)
                
                indices = np.random.choice(
                    len(songs_for_selection), 
                    size=min(remix_size, len(songs_for_selection)), 
                    replace=False,
                    p=probabilities
                )
                selected = [songs_for_selection[idx] for idx in indices]
            
            # Structure the remix
            if len(selected) == 3:
                remix = {
                    'intro': selected[0][0],
                    'build_up': selected[1][0],
                    'main_drop': selected[2][0],
                    'outro': selected[0][0],  # Loop back to intro
                    'avg_score': np.mean([s[1] for s in selected])
                }
            else:  # 4 songs
                remix = {
                    'intro': selected[0][0],
                    'build_up': selected[1][0],
                    'main_drop': selected[2][0],
                    'outro': selected[3][0],
                    'avg_score': np.mean([s[1] for s in selected])
                }
            
            remix['remix_id'] = i + 1
            remix['total_tracks'] = len([v for k, v in remix.items() if k in ['intro', 'build_up', 'main_drop', 'outro']])
            
            remixes.append(remix)
        
        return remixes
    
    def generate_remixes_for_song(self, song_name, num_remixes=7):
        """
        Main method to generate remix playlists for a song
        
        Args:
            song_name: Song to generate remixes for
            num_remixes: Number of remix variants (default: 7-8)
            
        Returns:
            List of remix structures with song data
        """
        # Step 1: Get associated songs
        associated_songs = self.get_associated_songs(song_name, top_n=12)
        
        if not associated_songs:
            return []
        
        # Step 2: Create remix structures
        remixes = self.create_remix_structure(associated_songs, num_remixes)
        
        # Step 3: Enrich with song data
        enriched_remixes = []
        for remix in remixes:
            enriched_remix = {
                'remix_id': remix['remix_id'],
                'total_tracks': remix['total_tracks'],
                'avg_score': remix['avg_score'],
                'tracks': {}
            }
            
            for position in ['intro', 'build_up', 'main_drop', 'outro']:
                if position in remix:
                    song_data = self.df[self.df['track_name'] == remix[position]]
                    if not song_data.empty:
                        enriched_remix['tracks'][position] = {
                            'track_name': remix[position],
                            'artist_name': song_data.iloc[0]['artist_name'],
                            'artwork_url': song_data.iloc[0].get('artwork_url', ''),
                            'track_preview_url': song_data.iloc[0].get('track_preview_url', ''),
                            'language': song_data.iloc[0].get('language', 'Unknown')
                        }
            
            if enriched_remix['tracks']:  # Only add if we have track data
                enriched_remixes.append(enriched_remix)
        
        return enriched_remixes


def initialize_remix_generator(df):
    """Initialize and prepare the remix generator"""
    generator = RemixGenerator(df)
    generator.create_transactions()
    generator.mine_associations()
    return generator


if __name__ == "__main__":
    # Example usage
    csv_path = "spotify_tracks_processed.csv"
    
    print("Loading data...")
    df = pd.read_csv(csv_path)
    
    print("Initializing Remix Generator...")
    generator = initialize_remix_generator(df)
    
    print("\nTesting remix generation...")
    test_song = input("Enter a song name: ")
    
    remixes = generator.generate_remixes_for_song(test_song, num_remixes=7)
    
    print(f"\nGenerated {len(remixes)} remixes:")
    for remix in remixes:
        print(f"\n--- Remix {remix['remix_id']} (Score: {remix['avg_score']:.2f}) ---")
        for position, track in remix['tracks'].items():
            print(f"  {position.upper()}: {track['track_name']} by {track['artist_name']}")
