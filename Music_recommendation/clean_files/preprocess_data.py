"""
preprocess_data.py
Run this script ONCE to preprocess your data and apply fuzzy logic
This will create a processed CSV file that the app can load in
stantly
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import pickle
import warnings
warnings.filterwarnings('ignore')


def preprocess_dataset(input_csv, output_csv):
    """Preprocess the dataset and save it"""
    print("="*60)
    print("STEP 1: Loading and Cleaning Data")
    print("="*60)
    
    # Load data
    df = pd.read_csv(input_csv)
    print(f"Initial dataset shape: {df.shape}")
    
    # Remove duplicates
    initial_rows = len(df)
    df.drop_duplicates(inplace=True)
    print(f"✓ Removed {initial_rows - len(df)} duplicate rows")
    
    # Replace -1 with NaN
    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols] = df[num_cols].replace(-1, np.nan)
    
    # Remove rows with missing values
    initial_rows = len(df)
    df.dropna(inplace=True)
    print(f"✓ Removed {initial_rows - len(df)} rows with missing values")
    
    # Normalize features
    features_to_normalize = ['popularity', 'loudness', 'tempo']
    scaler = MinMaxScaler()
    df[features_to_normalize] = scaler.fit_transform(df[features_to_normalize])
    print(f"✓ Normalized features: {features_to_normalize}")
    
    print(f"Final dataset shape: {df.shape}")
    
    return df


def create_fuzzy_system():
    """Create the fuzzy logic system"""
    print("\n" + "="*60)
    print("STEP 2: Creating Fuzzy Logic System")
    print("="*60)
    
    # Create Antecedents
    energy = ctrl.Antecedent(np.arange(0, 100.1, 0.1), 'energy')
    valence = ctrl.Antecedent(np.arange(0, 100.1, 0.1), 'valence')
    danceability = ctrl.Antecedent(np.arange(0, 100.1, 0.1), 'danceability')
    tempo = ctrl.Antecedent(np.arange(0, 250.1, 0.1), 'tempo')
    liveness = ctrl.Antecedent(np.arange(0, 100.1, 0.1), 'liveness')
    
    # Create Consequent
    mood = ctrl.Consequent(np.arange(0, 100.1, 0.1), 'mood')
    
    # Define membership functions
    energy['low'] = fuzz.trimf(energy.universe, [0, 0, 50])
    energy['medium'] = fuzz.trimf(energy.universe, [25, 50, 75])
    energy['high'] = fuzz.trimf(energy.universe, [50, 100, 100])
    
    valence['low'] = fuzz.trimf(valence.universe, [0, 0, 50])
    valence['medium'] = fuzz.trimf(valence.universe, [25, 50, 75])
    valence['high'] = fuzz.trimf(valence.universe, [50, 100, 100])
    
    danceability['low'] = fuzz.trimf(danceability.universe, [0, 0, 40])
    danceability['medium'] = fuzz.trimf(danceability.universe, [20, 50, 80])
    danceability['high'] = fuzz.trimf(danceability.universe, [60, 100, 100])
    
    tempo['low'] = fuzz.trimf(tempo.universe, [0, 0, 80])
    tempo['medium'] = fuzz.trimf(tempo.universe, [60, 120, 180])
    tempo['high'] = fuzz.trimf(tempo.universe, [150, 250, 250])
    
    liveness['low'] = fuzz.trimf(liveness.universe, [0, 0, 40])
    liveness['medium'] = fuzz.trimf(liveness.universe, [20, 50, 80])
    liveness['high'] = fuzz.trimf(liveness.universe, [60, 100, 100])
    
    mood['Sad'] = fuzz.trimf(mood.universe, [0, 0, 20])
    mood['Silent'] = fuzz.trimf(mood.universe, [20, 30, 40])
    mood['Chill'] = fuzz.trimf(mood.universe, [40, 50, 60])
    mood['Happy'] = fuzz.trimf(mood.universe, [60, 70, 80])
    mood['Workout'] = fuzz.trimf(mood.universe, [80, 100, 100])
    
    # Define rules
    rules = [
        ctrl.Rule(energy['low'] & valence['low'], mood['Sad']),
        ctrl.Rule(energy['low'] & valence['medium'] & liveness['low'], mood['Silent']),
        ctrl.Rule(energy['medium'] & valence['medium'], mood['Chill']),
        ctrl.Rule(valence['high'] & energy['medium'], mood['Happy']),
        ctrl.Rule(tempo['high'] & energy['high'] & danceability['high'], mood['Workout']),
        ctrl.Rule(liveness['high'] & valence['low'], mood['Chill']),
        ctrl.Rule(tempo['low'] & valence['low'], mood['Sad']),
        ctrl.Rule(energy['high'] & valence['high'], mood['Workout']),
        ctrl.Rule(energy['low'] & valence['high'], mood['Chill']),
        ctrl.Rule(energy['high'] & valence['low'], mood['Sad']),
        ctrl.Rule(danceability['high'] & energy['medium'], mood['Happy']),
        ctrl.Rule(danceability['low'] & energy['low'], mood['Sad']),
        ctrl.Rule(tempo['medium'] & energy['medium'] & valence['medium'], mood['Chill']),
        ctrl.Rule(liveness['medium'] & tempo['medium'], mood['Chill']),
        ctrl.Rule(liveness['high'] & energy['high'], mood['Workout']),
        ctrl.Rule(valence['low'] & danceability['low'], mood['Sad']),
        ctrl.Rule(tempo['high'] & energy['medium'], mood['Happy']),
        ctrl.Rule(energy['low'] & liveness['high'], mood['Silent']),
        ctrl.Rule(valence['high'] & liveness['high'], mood['Workout'])
    ]
    
    mood_ctrl = ctrl.ControlSystem(rules)
    mood_simulation = ctrl.ControlSystemSimulation(mood_ctrl)
    
    print("✓ Fuzzy logic system created")
    return mood_simulation


def classify_moods(df, mood_simulation):
    """Apply fuzzy logic to classify moods"""
    print("\n" + "="*60)
    print("STEP 3: Classifying Song Moods (This may take a few minutes...)")
    print("="*60)
    
    fuzzy_input_features = ['energy', 'valence', 'danceability', 'tempo', 'liveness']
    feature_ranges = {
        'energy': (0, 100),
        'valence': (0, 100),
        'danceability': (0, 100),
        'tempo': (0, 250),
        'liveness': (0, 100)
    }
    
    df['fuzzy_mood_score'] = np.nan
    total_songs = len(df)
    
    for index, row in df.iterrows():
        if index % 1000 == 0:
            print(f"Progress: {index}/{total_songs} songs processed ({index/total_songs*100:.1f}%)")
        
        valid_row = True
        for feature in fuzzy_input_features:
            min_bound, max_bound = feature_ranges[feature]
            scaled_value = row[feature] * (max_bound - min_bound) + min_bound
            
            if isinstance(scaled_value, (int, float)) and pd.notna(scaled_value):
                mood_simulation.input[feature] = scaled_value
            else:
                valid_row = False
                break
        
        if valid_row:
            try:
                mood_simulation.compute()
                df.loc[index, 'fuzzy_mood_score'] = mood_simulation.output['mood']
            except:
                df.loc[index, 'fuzzy_mood_score'] = np.nan
    
    # Assign mood labels
    def assign_mood_label(score):
        if pd.isna(score):
            return 'Unknown'
        elif 0 <= score < 20:
            return 'Sad'
        elif 20 <= score < 40:
            return 'Silent'
        elif 40 <= score < 60:
            return 'Chill'
        elif 60 <= score < 80:
            return 'Happy'
        elif 80 <= score <= 100:
            return 'Workout'
        else:
            return 'Unknown'
    
    df['mood_label'] = df['fuzzy_mood_score'].apply(assign_mood_label)
    
    print(f"\n✓ Mood classification complete!")
    print("\nMood Distribution:")
    print(df['mood_label'].value_counts())
    
    return df


def train_model(df):
    """Train the popularity prediction model"""
    print("\n" + "="*60)
    print("STEP 4: Training Popularity Prediction Model")
    print("="*60)
    
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    features = [col for col in numerical_cols if col not in 
                ['track_id', 'year', 'key', 'mode', 'time_signature', 'popularity']]
    
    X = df[features]
    y = df['popularity']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    from sklearn.metrics import mean_squared_error, r2_score
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"✓ Model trained successfully!")
    print(f"  Mean Squared Error: {mse:.4f}")
    print(f"  R-squared: {r2:.4f}")
    
    # Save model and scaler
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('features.pkl', 'wb') as f:
        pickle.dump(features, f)
    
    print("✓ Model, scaler, and features saved to disk")
    
    return model, scaler, features


def main():
    """Main preprocessing pipeline"""
    print("\n" + "🎵"*30)
    print("REVERB MUSIC RECOMMENDATION - DATA PREPROCESSING")
    print("🎵"*30 + "\n")
    
    INPUT_CSV = "spotify_tracks.csv"
    OUTPUT_CSV = "spotify_tracks_processed.csv"
    
    # Step 1: Clean and preprocess data
    df = preprocess_dataset(INPUT_CSV, OUTPUT_CSV)
    
    # Step 2: Create fuzzy system
    mood_simulation = create_fuzzy_system()
    
    # Step 3: Classify moods
    df = classify_moods(df, mood_simulation)
    
    # Step 4: Train model
    model, scaler, features = train_model(df)
    
    # Step 5: Save processed data
    print("\n" + "="*60)
    print("STEP 5: Saving Processed Data")
    print("="*60)
    
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✓ Processed data saved to: {OUTPUT_CSV}")
    
    print("\n" + "="*60)
    print("✅ PREPROCESSING COMPLETE!")
    print("="*60)
    print(f"\nFiles created:")
    print(f"  1. {OUTPUT_CSV} - Processed dataset with mood labels")
    print(f"  2. model.pkl - Trained XGBoost model")
    print(f"  3. scaler.pkl - Feature scaler")
    print(f"  4. features.pkl - Feature list")
    print(f"\nYou can now run: streamlit run app.py")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()