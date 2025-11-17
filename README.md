🎵 Reverb – A Hybrid Music Recommendation System

Reverb is an advanced hybrid music recommendation system that brings together multiple AI and analytical techniques—including KNN, DBSCAN Clustering, Mamdani Fuzzy Logic, Linear Regression, and Apriori Association-Rule Mining—to deliver context-aware, diverse, and interactive music recommendations.

Built with a modular architecture and deployed through an intuitive Streamlit interface, Reverb helps users explore:
✔ language-based playlists
✔ mood-conditioned recommendations
✔ remix songs
✔ similar song discovery
✔ popularity-based ranking and analysis
✔ roulette-wheel random selection

 Features
 1. Similar Song Recommendation

Computes similarity using Cosine Similarity and KNN.

Finds musically coherent neighbors based on acoustic features.

 2. Language-Based Playlist Generation

Uses DBSCAN to detect natural clusters in multilingual Spotify data.

Generates playlists for English, Hindi, Tamil, Telugu, Malayalam, Korean.

 3. Mood Classification (Fuzzy Logic)

Mamdani Fuzzy Inference System using:
Energy
Valence
Danceability
Tempo
Liveness
Produces:
Continuous mood score (0–100)
Mood label (Sad, Silent, Chill, Happy, Workout)

 4. Popularity Prediction
Predicts track popularity using a Linear Regression model.

Helps in hybrid ranking for artist-focused recommendations.

5. Remix Playlist Generator (Apriori Algorithm)

Discovers frequent co-occurrence patterns between tracks.

Creates structured remix playlists with:

Intro

Build-Up

Main Drop

Outro

 6. Spin-Wheel Discovery Module
Engaging roulette-wheel selector for random song discovery.
Adds interactive exploration to the system.

 Dataset

Source: Kaggle – Spotify Tracks Dataset
https://www.kaggle.com/datasets/gauthamvijayaraj/spotify-tracks-dataset-updated-every-week
~60,000 tracks with:
Audio features
Metadata (track name, artist, album, year, language)
Popularity scores
Preprocessing Steps:
Duplicate removal
Handling invalid/missing values
MinMax Scaling (popularity, loudness, tempo)
Standardization for regression
Feature scaling for fuzzy mood inputs
Output file: Spotify_tracks_processed.csv

 System Architecture
1. Preprocessing Layer
Cleans, normalizes, and prepares enriched features.
2. Mood Classification
Generates fuzzy mood scores and labels.
3. Similarity & Clustering Engine
Handles similarity search + DBSCAN clustering for language playlists.
4. Predictive Modeling
Linear Regression predicts popularity scores.
5. Association-Rule Mining
Apriori algorithm mines frequent itemsets and rules.
6. Hybrid Recommendation Layer
Combines popularity + similarity + mood filtering.
7. Roulette-Wheel Selector
Adds randomization to user experience.
8. Streamlit User Interface

Search

Recommendations

Audio player

Remix cards

Spin wheel

Methodology Overview
 Similarity Modeling

Cosine, Euclidean, KNN, K-Means similarity, Weighted Cosine
→ Final choice: KNN

 Language Clustering

Heuristic, K-Means, DBSCAN
→ Final choice: DBSCAN

Regression Models

Linear Regression, Random Forest, XGBoost
→ Final choice: Linear Regression

Mood Classification

Mamdani fuzzy inference system with 19 rules

Association Rules
Apriori algorithm with support, confidence, lift thresholds

Evaluation & Results
Similar Song Recommendation

KNN produced the most musically coherent results.

Language Clustering

DBSCAN showed highest purity and best noise handling.
Popularity Prediction

Linear Regression achieved lowest MSE & fastest inference.

Mood Classification
Smooth, interpretable mood transitions.

Popularity Analyis:
Based on era 
mood based playlists
language based playlists
popular artists of all time


Remix Generation
Apriori rules produced meaningful and diverse remix structures.

Spin-Wheel Evaluation
Fair probability distribution
Increased user engagement

 Conclusion

Reverb demonstrates how hybrid AI systems create richer, more personalized music recommendations. By combining:

Similarity modeling

Clustering

Fuzzy logic

Regression

Association-rule mining

Interactive UI

the system achieves greater diversity, interpretability, and user engagement than single-method recommenders.

Tech Stack
Python
Streamlit
Scikit-learn
Pandas
Numpy
Matplotlib/Seaborn
Fuzzy Logic Toolbox (skfuzzy)
MLxtend for Apriori
