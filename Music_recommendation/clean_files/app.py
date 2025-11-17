"""
app.py
Streamlit Web Application for Reverb Music Recommendation System
Updated with Remix Playlists Feature, Spin Wheel, and Analytics Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
from recommender import MusicRecommender
import os
import time
import base64
import random
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Reverb - Music Recommendation",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session state for currently playing song
if 'current_song' not in st.session_state:
    st.session_state.current_song = None
if 'is_playing' not in st.session_state:
    st.session_state.is_playing = False
if 'selected_song_data' not in st.session_state:
    st.session_state.selected_song_data = None
if 'spin_result' not in st.session_state:
    st.session_state.spin_result = None
if 'show_spin_wheel' not in st.session_state:
    st.session_state.show_spin_wheel = False

# Custom CSS for dark purple theme matching the design
st.markdown("""
<style>
    /* Main app styling */
    .stApp {
        background-color: #1a0b2e;
    }
    
    /* Header styling */
    .main-header {
        display: flex;
        align-items: center;
        margin-bottom: 20px;
    }
    
    .logo {
        font-size: 48px;
        margin-right: 15px;
    }
    
    .app-name {
        font-size: 48px;
        font-weight: bold;
        color: white;
    }
    
    /* Music Player Styling */
    .music-player {
        background: linear-gradient(135deg, #2d1b4e 0%, #1a0b2e 100%);
        border-radius: 20px;
        padding: 25px;
        margin: 20px 0;
        border: 2px solid #9b7fd4;
        box-shadow: 0 8px 32px rgba(155, 127, 212, 0.3);
    }
    
    .player-content {
        display: flex;
        align-items: center;
        gap: 25px;
    }
    
    .album-art {
        width: 140px;
        height: 140px;
        border-radius: 15px;
        object-fit: cover;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.5);
        flex-shrink: 0;
    }
    
    .player-info {
        flex: 1;
    }
    
    .now-playing-label {
        color: #9b7fd4;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 1px;
        margin-bottom: 5px;
    }
    
    .player-track-name {
        font-size: 28px;
        font-weight: bold;
        color: white;
        margin: 5px 0;
    }
    
    .player-artist {
        font-size: 18px;
        color: rgba(255, 255, 255, 0.8);
        margin-bottom: 20px;
    }
    
    /* Search bar styling */
    .stTextInput input {
        background-color: #e8e5f0;
        border-radius: 30px;
        padding: 15px 25px;
        border: none;
        font-size: 16px;
        color: #4b367c;
    }
    
    /* Section headers */
    .section-header {
        background-color: #9b7fd4;
        color: white;
        padding: 10px 25px;
        border-radius: 25px;
        font-size: 18px;
        font-weight: 600;
        display: inline-block;
        margin: 20px 0 15px 0;
    }
    
    /* Song cards */
    .song-card {
        background-color: #2d1b4e;
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        color: white;
        transition: transform 0.2s;
        cursor: pointer;
    }
    
    .song-card:hover {
        transform: translateY(-5px);
        background-color: #3d2b5e;
    }
    
    .song-card-with-image {
        display: flex;
        align-items: center;
        gap: 15px;
    }
    
    .song-thumbnail {
        width: 60px;
        height: 60px;
        border-radius: 8px;
        object-fit: cover;
        flex-shrink: 0;
    }
    
    .song-info {
        flex: 1;
        min-width: 0;
    }
    
    /* Remix card styling */
    .remix-card {
        background: linear-gradient(135deg, #2d1b4e 0%, #1a0b2e 100%);
        border-radius: 20px;
        padding: 25px;
        margin: 15px 0;
        border: 2px solid #ff6b9d;
        box-shadow: 0 8px 25px rgba(255, 107, 157, 0.3);
        color: white;
        transition: transform 0.2s;
    }
    
    .remix-card:hover {
        transform: translateY(-3px);
        border-color: #ff8fb3;
    }
    
    .remix-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 20px;
    }
    
    .remix-title {
        font-size: 22px;
        font-weight: bold;
        color: #ff6b9d;
    }
    
    .remix-badge {
        background-color: #ff6b9d;
        color: white;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }
    
    .remix-track {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 15px;
        margin: 10px 0;
        display: flex;
        align-items: center;
        gap: 15px;
        border-left: 4px solid #ff6b9d;
    }
    
    .remix-position {
        background-color: #ff6b9d;
        color: white;
        padding: 8px 12px;
        border-radius: 8px;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        min-width: 90px;
        text-align: center;
    }
    
    .remix-track-info {
        flex: 1;
    }
    
    .remix-track-name {
        font-size: 15px;
        font-weight: 600;
        margin-bottom: 5px;
    }
    
    .remix-track-artist {
        font-size: 13px;
        opacity: 0.7;
    }
    
    /* Currently playing card */
    .playing-card {
        background-color: #2d1b4e;
        border-radius: 20px;
        padding: 30px;
        margin: 20px 0;
        color: white;
        border: 2px solid #9b7fd4;
    }
    
    .playing-card-image {
        width: 100%;
        max-width: 300px;
        border-radius: 15px;
        margin-bottom: 20px;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.5);
    }
    
    /* Small card for playlists */
    .small-card {
        background-color: #2d1b4e;
        border-radius: 12px;
        padding: 15px;
        margin: 8px 0;
        color: white;
        font-size: 14px;
        cursor: pointer;
        transition: transform 0.2s;
    }
    
    .small-card:hover {
        transform: translateX(5px);
        background-color: #3d2b5e;
    }
    
    .small-card-with-image {
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .small-thumbnail {
        width: 40px;
        height: 40px;
        border-radius: 6px;
        object-fit: cover;
        flex-shrink: 0;
    }
    
    /* Language/Mood buttons */
    .category-button {
        background-color: #9b7fd4;
        color: white;
        padding: 8px 20px;
        border-radius: 20px;
        font-size: 16px;
        font-weight: 600;
        text-align: center;
        margin-bottom: 10px;
    }
    
    /* Text colors */
    h1, h2, h3, h4, h5, h6, p, label {
        color: white !important;
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Column spacing */
    div[data-testid="column"] {
        padding: 10px;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background-color: #9b7fd4;
    }
    
    /* Hide default audio player */
    audio {
        display: none;
    }
    
    /* Spin Wheel Button Styling */
    .spin-wheel-button {
        background: linear-gradient(135deg, #ff6b9d 0%, #9b7fd4 100%);
        color: white;
        padding: 15px 30px;
        border-radius: 30px;
        font-size: 18px;
        font-weight: 600;
        text-align: center;
        cursor: pointer;
        border: none;
        box-shadow: 0 8px 20px rgba(255, 107, 157, 0.4);
        transition: transform 0.2s;
        display: inline-block;
        margin: 20px 0;
    }
    
    .spin-wheel-button:hover {
        transform: scale(1.05);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_recommender(csv_path):
    """Load and initialize the recommender system (FAST - uses preprocessed data)"""
    
    status_text = st.empty()
    
    try:
        status_text.text("⚡ Loading preprocessed data...")
        recommender = MusicRecommender(csv_path)
        recommender.load_preprocessed_data()
        
        status_text.text("✅ Ready!")
        import time
        time.sleep(0.3)
        
        # Clear status
        status_text.empty()
        
        return recommender
        
    except Exception as e:
        status_text.empty()
        raise e


def display_music_player(song_data):
    """Display the music player with custom controls"""
    album_art = song_data.get('artwork_url', '')
    preview_url = song_data.get('track_preview_url', '')
    #preview_url=st.markdown(display_small_card(song_data), unsafe_allow_html=True)
    
    # Default image if none available
    if not album_art or pd.isna(album_art) or album_art == '':
        album_art = "https://via.placeholder.com/140x140/9b7fd4/ffffff?text=No+Image"
    
    # Escape special characters in song data
    track_name = str(song_data['track_name']).replace('"', '&quot;').replace("'", "&#39;")
    artist_name = str(song_data['artist_name']).replace('"', '&quot;').replace("'", "&#39;")
    
    # Generate unique ID for this audio player
    audio_id = "audio_player"
    
    # Use components.html for better HTML rendering
    import streamlit.components.v1 as components
    
    player_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            .music-player {{
                background: linear-gradient(135deg, #2d1b4e 0%, #1a0b2e 100%);
                border-radius: 20px;
                padding: 25px;
                margin: 20px 0;
                border: 2px solid #9b7fd4;
                box-shadow: 0 8px 32px rgba(155, 127, 212, 0.3);
            }}
            
            .player-content {{
                display: flex;
                align-items: center;
                gap: 25px;
            }}
            
            .album-art {{
                width: 140px;
                height: 140px;
                border-radius: 15px;
                object-fit: cover;
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.5);
                flex-shrink: 0;
            }}
            
            .player-info {{
                flex: 1;
            }}
            
            .now-playing-label {{
                color: #9b7fd4;
                font-size: 12px;
                font-weight: 600;
                letter-spacing: 1px;
                margin-bottom: 5px;
            }}
            
            .player-track-name {{
                font-size: 28px;
                font-weight: bold;
                color: white;
                margin: 5px 0;
            }}
            
            .player-artist {{
                font-size: 18px;
                color: rgba(255, 255, 255, 0.8);
                margin-bottom: 20px;
            }}
            
            .player-controls {{
                display: flex;
                align-items: center;
                gap: 20px;
                margin-top: 20px;
            }}
            
            .control-button {{
                background-color: #9b7fd4;
                color: white;
                border: none;
                border-radius: 50%;
                width: 50px;
                height: 50px;
                font-size: 20px;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                transition: background-color 0.2s;
            }}
            
            .control-button:hover {{
                background-color: #7a5fb8;
            }}
            
            .progress-container {{
                flex: 1;
            }}
            
            .time-labels {{
                display: flex;
                justify-content: space-between;
                font-size: 12px;
                color: rgba(255, 255, 255, 0.6);
                margin-top: 5px;
            }}
            
            .volume-container {{
                display: flex;
                align-items: center;
                gap: 10px;
                min-width: 150px;
            }}
            
            .volume-icon {{
                color: #9b7fd4;
                font-size: 20px;
            }}
            
            input[type="range"] {{
                -webkit-appearance: none;
                width: 100%;
                height: 6px;
                border-radius: 3px;
                background: #3d2b5e;
                outline: none;
            }}
            
            input[type="range"]::-webkit-slider-thumb {{
                -webkit-appearance: none;
                appearance: none;
                width: 16px;
                height: 16px;
                border-radius: 50%;
                background: #9b7fd4;
                cursor: pointer;
            }}
            
            input[type="range"]::-moz-range-thumb {{
                width: 16px;
                height: 16px;
                border-radius: 50%;
                background: #9b7fd4;
                cursor: pointer;
                border: none;
            }}
        </style>
    </head>
    <body>
        <div class="music-player">
            <div class="player-content">
                <img src="{album_art}" alt="Album Art" class="album-art" onerror="this.src='https://via.placeholder.com/140x140/9b7fd4/ffffff?text=No+Image'">
                <div class="player-info">
                    <div class="now-playing-label">🎵 NOW PLAYING</div>
                    <div class="player-track-name">{track_name}</div>
                    <div class="player-artist">{artist_name}</div>
                    
                    <div class="player-controls">
                        <button class="control-button" onclick="togglePlay()" id="playPauseBtn">
                            ▶️
                        </button>
                        
                        <div class="progress-container">
                            <input type="range" min="0" max="100" value="0" id="progressBar" 
                                   onchange="seekAudio(this.value)">
                            <div class="time-labels">
                                <span id="currentTime">0:00</span>
                                <span id="duration">0:00</span>
                            </div>
                        </div>
                        
                        <div class="volume-container">
                            <span class="volume-icon">🔊</span>
                            <input type="range" min="0" max="100" value="70" id="volumeBar" 
                                   onchange="changeVolume(this.value)">
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <audio id="{audio_id}" src="{preview_url if preview_url and not pd.isna(preview_url) else ''}"
               ontimeupdate="updateProgress()" 
               onloadedmetadata="updateDuration()"
               onended="onAudioEnd()">
        </audio>
        
        <script>
            const audio = document.getElementById('{audio_id}');
            const playPauseBtn = document.getElementById('playPauseBtn');
            const progressBar = document.getElementById('progressBar');
            const volumeBar = document.getElementById('volumeBar');
            const currentTimeLabel = document.getElementById('currentTime');
            const durationLabel = document.getElementById('duration');
            
            // Set initial volume
            audio.volume = 0.7;
            
            function togglePlay() {{
                if (audio.paused) {{
                    audio.play();
                    playPauseBtn.innerHTML = '⏸️';
                }} else {{
                    audio.pause();
                    playPauseBtn.innerHTML = '▶️';
                }}
            }}
            
            function seekAudio(value) {{
                const time = (value / 100) * audio.duration;
                audio.currentTime = time;
            }}
            
            function changeVolume(value) {{
                audio.volume = value / 100;
            }}
            
            function updateProgress() {{
                if (audio.duration) {{
                    const progress = (audio.currentTime / audio.duration) * 100;
                    progressBar.value = progress;
                    currentTimeLabel.textContent = formatTime(audio.currentTime);
                }}
            }}
            
            function updateDuration() {{
                if (audio.duration) {{
                    durationLabel.textContent = formatTime(audio.duration);
                }}
            }}
            
            function formatTime(seconds) {{
                const mins = Math.floor(seconds / 60);
                const secs = Math.floor(seconds % 60);
                return mins + ':' + (secs < 10 ? '0' : '') + secs;
            }}
            
            function onAudioEnd() {{
                playPauseBtn.innerHTML = '▶️';
                progressBar.value = 0;
            }}
            
            // Auto-play if audio source is available
            if (audio.src && audio.src !== '') {{
                audio.play().then(() => {{
                    playPauseBtn.innerHTML = '⏸️';
                }}).catch(e => {{
                    console.log('Auto-play prevented:', e);
                }});
            }}
        </script>
    </body>
    </html>
    """
    
    components.html(player_html, height=250)
    
    # Show message if no preview available
    if not preview_url or pd.isna(preview_url) or preview_url == '':
        st.warning("⚠️ No audio preview available for this track")


def display_spin_wheel(languages, recommender):
    """Display the fortune wheel for song selection"""
    import streamlit.components.v1 as components
    
    # Get random songs from selected languages
    songs_pool = recommender.df[recommender.df['language'].isin(languages)].copy()
    
    if len(songs_pool) < 8:
        st.error("Not enough songs available in selected languages!")
        return None
    
    # Select 8 random songs for the wheel
    wheel_songs = songs_pool.sample(8).to_dict('records')
    
    # Prepare song data for JavaScript
    songs_json = []
    for idx, song in enumerate(wheel_songs):
        songs_json.append({
            'text': f"{song['track_name'][:25]}..." if len(song['track_name']) > 25 else song['track_name'],
            'full_name': song['track_name'],
            'artist': song['artist_name'],
            'color': ['#ff6b9d', '#9b7fd4', '#ff8fb3', '#b094d4', '#ff4d7d', '#8b6fc4', '#ff9db3', '#7a5fb8'][idx]
        })
    
    wheel_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{
                margin: 0;
                padding: 20px;
                background-color: #1a0b2e;
                font-family: Arial, sans-serif;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
            }}
            
            .wheel-container {{
                position: relative;
                width: 500px;
                height: 500px;
                margin: 20px auto;
            }}
            
            #canvas {{
                position: absolute;
                top: 0;
                left: 0;
            }}
            
            .pointer {{
                position: absolute;
                top: -20px;
                left: 50%;
                transform: translateX(-50%);
                width: 0;
                height: 0;
                border-left: 20px solid transparent;
                border-right: 20px solid transparent;
                border-top: 40px solid #ff6b9d;
                z-index: 10;
                filter: drop-shadow(0 4px 8px rgba(0,0,0,0.5));
            }}
            
            .spin-button {{
                background: linear-gradient(135deg, #ff6b9d 0%, #9b7fd4 100%);
                color: white;
                padding: 20px 50px;
                border-radius: 50px;
                font-size: 24px;
                font-weight: bold;
                border: none;
                cursor: pointer;
                margin-top: 30px;
                box-shadow: 0 8px 20px rgba(255, 107, 157, 0.4);
                transition: transform 0.2s;
            }}
            
            .spin-button:hover {{
                transform: scale(1.05);
            }}
            
            .spin-button:disabled {{
                opacity: 0.5;
                cursor: not-allowed;
            }}
            
            .result-display {{
                margin-top: 30px;
                padding: 30px;
                background: linear-gradient(135deg, #2d1b4e 0%, #1a0b2e 100%);
                border-radius: 20px;
                border: 2px solid #ff6b9d;
                color: white;
                text-align: center;
                min-height: 150px;
                display: none;
                box-shadow: 0 8px 25px rgba(255, 107, 157, 0.3);
            }}
            
            .result-display.show {{
                display: block;
                animation: fadeIn 0.5s;
            }}
            
            @keyframes fadeIn {{
                from {{ opacity: 0; transform: translateY(-20px); }}
                to {{ opacity: 1; transform: translateY(0); }}
            }}
            
            .winner-text {{
                font-size: 28px;
                font-weight: bold;
                color: #ff6b9d;
                margin-bottom: 15px;
            }}
            
            .song-name {{
                font-size: 24px;
                font-weight: bold;
                margin-bottom: 10px;
            }}
            
            .artist-name {{
                font-size: 18px;
                opacity: 0.8;
            }}
        </style>
    </head>
    <body>
        <div class="wheel-container">
            <div class="pointer"></div>
            <canvas id="canvas" width="500" height="500"></canvas>
        </div>
        
        <button class="spin-button" id="spinBtn" onclick="spin()">🎰 SPIN THE WHEEL!</button>
        
        <div class="result-display" id="result">
            <div class="winner-text">🎉 YOUR SONG IS...</div>
            <div class="song-name" id="winnerSong"></div>
            <div class="artist-name" id="winnerArtist"></div>
        </div>
        
        <script>
            const songs = {songs_json};
            const canvas = document.getElementById('canvas');
            const ctx = canvas.getContext('2d');
            const centerX = canvas.width / 2;
            const centerY = canvas.height / 2;
            const radius = 220;
            
            let currentAngle = 0;
            let spinning = false;
            
            function drawWheel() {{
                const sliceAngle = (2 * Math.PI) / songs.length;
                
                for (let i = 0; i < songs.length; i++) {{
                    const startAngle = currentAngle + i * sliceAngle;
                    const endAngle = startAngle + sliceAngle;
                    
                    // Draw slice
                    ctx.beginPath();
                    ctx.moveTo(centerX, centerY);
                    ctx.arc(centerX, centerY, radius, startAngle, endAngle);
                    ctx.closePath();
                    ctx.fillStyle = songs[i].color;
                    ctx.fill();
                    ctx.strokeStyle = '#1a0b2e';
                    ctx.lineWidth = 3;
                    ctx.stroke();
                    
                    // Draw text
                    ctx.save();
                    ctx.translate(centerX, centerY);
                    ctx.rotate(startAngle + sliceAngle / 2);
                    ctx.textAlign = 'center';
                    ctx.fillStyle = 'white';
                    ctx.font = 'bold 14px Arial';
                    ctx.fillText(songs[i].text, radius * 0.65, 0);
                    ctx.restore();
                }}
                
                // Draw center circle
                ctx.beginPath();
                ctx.arc(centerX, centerY, 40, 0, 2 * Math.PI);
                ctx.fillStyle = '#ff6b9d';
                ctx.fill();
                ctx.strokeStyle = 'white';
                ctx.lineWidth = 3;
                ctx.stroke();
            }}
            
            function spin() {{
                if (spinning) return;
                
                spinning = true;
                document.getElementById('spinBtn').disabled = true;
                document.getElementById('result').classList.remove('show');
                
                const totalRotation = (Math.PI * 2) * 5 + (Math.random() * Math.PI * 2);
                const duration = 4000;
                const startTime = Date.now();
                
                function animate() {{
                    const elapsed = Date.now() - startTime;
                    const progress = Math.min(elapsed / duration, 1);
                    
                    // Easing function
                    const easeOut = 1 - Math.pow(1 - progress, 3);
                    currentAngle = totalRotation * easeOut;
                    
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    drawWheel();
                    
                    if (progress < 1) {{
                        requestAnimationFrame(animate);
                    }} else {{
                        spinning = false;
                        document.getElementById('spinBtn').disabled = false;
                        showResult();
                    }}
                }}
                
                animate();
            }}
            
            function showResult() {{
                // Calculate winner based on pointer position (top center)
                const normalizedAngle = (currentAngle % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI);
                const sliceAngle = (2 * Math.PI) / songs.length;
                
                // The pointer points down, so we need to check which slice is at the top
                // Add PI/2 to account for starting position and reverse direction
                const pointerAngle = (normalizedAngle+ Math.PI / 4 ) % (2 * Math.PI);
                //const pointerAngle = (2 * Math.PI - normalizedAngle - Math.PI / 2) % (2 * Math.PI);
                const winnerIndex = Math.floor(pointerAngle/sliceAngle ) % songs.length;
                const winner = songs[winnerIndex];
                
                // Display winner
                document.getElementById('winnerSong').textContent = winner.full_name;
                document.getElementById('winnerArtist').textContent = 'by ' + winner.artist;
                
                document.getElementById('result').classList.add('show');
                
                // Send winner back to Streamlit
                window.parent.postMessage({{
                    type: 'streamlit:setComponentValue',
                    value: winner.full_name
                }}, '*');
            }}
            
            // Initial draw
            drawWheel();
        </script>
    </body>
    </html>
    """
    
    result = components.html(wheel_html, height=900, scrolling=False)
    return result


def display_song_card(song_data, show_score=None, score_label="Score"):
    """Display a song card with thumbnail"""
    album_art = song_data.get('artwork_url', '')
    
    # Default image if none available
    if not album_art or pd.isna(album_art) or album_art == '':
        album_art = "https://via.placeholder.com/60x60/9b7fd4/ffffff?text=♪"
    
    score_text = ""
    if show_score is not None:
        if isinstance(show_score, float):
            if show_score < 1:
                score_text = f"<small>{score_label}: {show_score:.2%}</small>"
            else:
                score_text = f"<small>{score_label}: {show_score:.2f}</small>"
    
    card_html = f"""
    <div class="song-card song-card-with-image">
        <img src="{album_art}" alt="Album" class="song-thumbnail" onerror="this.src='https://via.placeholder.com/60x60/9b7fd4/ffffff?text=♪'">
        <div class="song-info">
            <strong style="font-size: 16px;">{song_data['track_name']}</strong><br>
            <small style="opacity: 0.8;">{song_data['artist_name']}</small>
            {f'<br>{score_text}' if score_text else ''}
        </div>
    </div>
    """
    return card_html


def display_small_card(song_data):
    """Display a small card for playlists with thumbnail"""
    album_art = song_data.get('artwork_url', '')
    
    # Default image if none available
    if not album_art or pd.isna(album_art) or album_art == '':
        album_art = "https://via.placeholder.com/40x40/9b7fd4/ffffff?text=♪"
    
    card_html = f"""
    <div class="small-card small-card-with-image">
        <img src="{album_art}" alt="Album" class="small-thumbnail" onerror="this.src='https://via.placeholder.com/40x40/9b7fd4/ffffff?text=♪'">
        <div>
            <strong style="font-size: 13px;">{song_data['track_name']}</strong><br>
            <small style="opacity: 0.7; font-size: 11px;">{song_data['artist_name']}</small>
        </div>
    </div>
    """
    return card_html


def display_remix_card(remix, remix_idx, recommender):
    """Display a remix card with all tracks"""
    remix_html = f"""
    <div class="remix-card">
        <div class="remix-header">
            <div class="remix-title">🎧 Remix #{remix['remix_id']}</div>
            <div class="remix-badge">{remix['total_tracks']} TRACKS • SCORE: {remix['avg_score']:.2f}</div>
        </div>
    """
    
    st.markdown(remix_html, unsafe_allow_html=True)
    
    # Display each track in the remix
    position_order = ['intro', 'build_up', 'main_drop', 'outro']
    position_labels = {
        'intro': '🎹 INTRO',
        'build_up': '📈 BUILD-UP',
        'main_drop': '💥 MAIN DROP',
        'outro': '🌙 OUTRO'
    }
    
    for position in position_order:
        if position in remix['tracks']:
            track = remix['tracks'][position]
            album_art = track.get('artwork_url', '')
            if not album_art or album_art == '':
                album_art = "https://via.placeholder.com/50x50/ff6b9d/ffffff?text=♪"
            
            track_html = f"""
            <div class="remix-track">
                <div class="remix-position">{position_labels[position]}</div>
                <img src="{album_art}" alt="Album" style="width: 50px; height: 50px; border-radius: 8px;" onerror="this.src='https://via.placeholder.com/50x50/ff6b9d/ffffff?text=♪'">
                <div class="remix-track-info">
                    <div class="remix-track-name">{track['track_name']}</div>
                    <div class="remix-track-artist">{track['artist_name']}</div>
                </div>
            </div>
            """
            st.markdown(track_html, unsafe_allow_html=True)
            
            # Play button for each track
            if st.button(f"▶️ Play", key=f"play_remix_{remix_idx}_{position}", use_container_width=True):
                # Find the full song data from recommender
                song_row = recommender.df[recommender.df['track_name'] == track['track_name']]
                if not song_row.empty:
                    st.session_state.selected_song_data = song_row.iloc[0]
                    st.rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)


def create_analytics_visualizations(recommender):
    """Create analytics visualizations for popular songs"""
    df = recommender.df.copy()
    
    # Define popularity threshold (songs with popularity > 0.5 are considered popular)
    popular_threshold = 0.5
    df['is_popular'] = df['popularity'] > popular_threshold
    
    # 1. Popular Songs by Language
    st.markdown('<div class="section-header">📊 Popular Songs by Language</div>', unsafe_allow_html=True)
    st.markdown('<p style="color: rgba(255, 255, 255, 0.7); font-size: 14px; margin-bottom: 20px;">Number of popular songs (popularity > 0.5) across different languages</p>', unsafe_allow_html=True)
    
    lang_popular = df[df['is_popular']].groupby('language').size().reset_index(name='count')
    lang_popular = lang_popular.sort_values('count', ascending=False).head(10)
    
    fig1 = px.bar(
        lang_popular,
        x='language',
        y='count',
        title='Top 10 Languages by Popular Song Count',
        labels={'language': 'Language', 'count': 'Number of Popular Songs'},
        color='count',
        color_continuous_scale=['#9b7fd4', '#ff6b9d']
    )
    
    fig1.update_layout(
        plot_bgcolor='#1a0b2e',
        paper_bgcolor='#1a0b2e',
        font_color='white',
        title_font_size=20,
        title_font_color='#9b7fd4',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='#2d1b4e'),
        showlegend=False
    )
    
    st.plotly_chart(fig1, use_container_width=True)
    
    # 2. Year-wise Popular Songs Analysis
    st.markdown('<div class="section-header">📅 Popular Songs by Era</div>', unsafe_allow_html=True)
    st.markdown('<p style="color: rgba(255, 255, 255, 0.7); font-size: 14px; margin-bottom: 20px;">Distribution of popular songs across different time periods</p>', unsafe_allow_html=True)
    
    # Check if year column exists
    if 'year' in df.columns:
        # Create era categories
        def categorize_year(year):
            if pd.isna(year) or year == 0:
                return 'Unknown'
            elif year < 1980:
                return 'Before 1980s'
            elif 1980 <= year < 1990:
                return '1980s'
            elif 1990 <= year < 2000:
                return '1990s'
            elif 2000 <= year < 2010:
                return '2000s'
            elif 2010 <= year < 2020:
                return '2010s'
            else:
                return '2020s'
        
        df['era'] = df['year'].apply(categorize_year)
        
        # Count popular songs by era
        era_popular = df[df['is_popular']].groupby('era').size().reset_index(name='count')
        
        # Order the eras properly
        era_order = ['Before 1980s', '1980s', '1990s', '2000s', '2010s', '2020s', 'Unknown']
        era_popular['era'] = pd.Categorical(era_popular['era'], categories=era_order, ordered=True)
        era_popular = era_popular.sort_values('era')
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Bar chart
            fig2 = px.bar(
                era_popular,
                x='era',
                y='count',
                title='Popular Songs by Era',
                labels={'era': 'Time Period', 'count': 'Number of Popular Songs'},
                color='count',
                color_continuous_scale=['#ff6b9d', '#9b7fd4']
            )
            
            fig2.update_layout(
                plot_bgcolor='#1a0b2e',
                paper_bgcolor='#1a0b2e',
                font_color='white',
                title_font_size=18,
                title_font_color='#ff6b9d',
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor='#2d1b4e'),
                showlegend=False
            )
            
            st.plotly_chart(fig2, use_container_width=True)
        
        with col2:
            # Pie chart
            fig3 = px.pie(
                era_popular,
                values='count',
                names='era',
                title='Distribution of Popular Songs by Era',
                color_discrete_sequence=['#ff6b9d', '#9b7fd4', '#ff8fb3', '#b094d4', '#ff4d7d', '#8b6fc4', '#7a5fb8']
            )
            
            fig3.update_layout(
                plot_bgcolor='#1a0b2e',
                paper_bgcolor='#1a0b2e',
                font_color='white',
                title_font_size=18,
                title_font_color='#ff6b9d'
            )
            
            st.plotly_chart(fig3, use_container_width=True)
        
        # 3. Detailed Year-wise Trends (for recent years)
        st.markdown('<div class="section-header">📈 Recent Trends (2015-2025)</div>', unsafe_allow_html=True)
        
        recent_years = df[(df['year'] >= 2015) & (df['year'] <= 2025) & (df['is_popular'])]
        if not recent_years.empty:
            year_counts = recent_years.groupby('year').size().reset_index(name='count')
            
            fig4 = px.line(
                year_counts,
                x='year',
                y='count',
                title='Popular Songs Trend (2015-2025)',
                labels={'year': 'Year', 'count': 'Number of Popular Songs'},
                markers=True
            )
            
            fig4.update_traces(
                line_color='#ff6b9d',
                marker=dict(size=10, color='#9b7fd4')
            )
            
            fig4.update_layout(
                plot_bgcolor='#1a0b2e',
                paper_bgcolor='#1a0b2e',
                font_color='white',
                title_font_size=18,
                title_font_color='#9b7fd4',
                xaxis=dict(showgrid=True, gridcolor='#2d1b4e'),
                yaxis=dict(showgrid=True, gridcolor='#2d1b4e')
            )
            
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.info("No recent year data available for popular songs")
    else:
        st.warning("Year information not available in the dataset")
    
    # 4. Mood-wise Popularity Analysis (Recent Years)
    st.markdown('<div class="section-header">🎭 Mood-wise Popularity Trends</div>', unsafe_allow_html=True)
    st.markdown('<p style="color: rgba(255, 255, 255, 0.7); font-size: 14px; margin-bottom: 20px;">Average popularity scores for different moods in recent years (2015-2025)</p>', unsafe_allow_html=True)
    
    if 'mood_label' in df.columns and 'year' in df.columns:
        # Filter for recent years (2015-2025)
        recent_mood_data = df[(df['year'] >= 2015) & (df['year'] <= 2025)].copy()
        
        if not recent_mood_data.empty and 'Unknown' in recent_mood_data['mood_label'].values:
            recent_mood_data = recent_mood_data[recent_mood_data['mood_label'] != 'Unknown']
        
        if not recent_mood_data.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # Average popularity by mood
                mood_popularity = recent_mood_data.groupby('mood_label')['popularity'].mean().reset_index()
                mood_popularity = mood_popularity.sort_values('popularity', ascending=False)
                
                fig5 = px.bar(
                    mood_popularity,
                    x='mood_label',
                    y='popularity',
                    title='Average Popularity by Mood (2015-2025)',
                    labels={'mood_label': 'Mood', 'popularity': 'Average Popularity Score'},
                    color='popularity',
                    color_continuous_scale=['#9b7fd4', '#ff6b9d']
                )
                
                fig5.update_layout(
                    plot_bgcolor='#1a0b2e',
                    paper_bgcolor='#1a0b2e',
                    font_color='white',
                    title_font_size=18,
                    title_font_color='#ff6b9d',
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=True, gridcolor='#2d1b4e'),
                    showlegend=False
                )
                
                st.plotly_chart(fig5, use_container_width=True)
            
            with col2:
                # Count of songs by mood
                mood_count = recent_mood_data.groupby('mood_label').size().reset_index(name='count')
                
                fig6 = px.pie(
                    mood_count,
                    values='count',
                    names='mood_label',
                    title='Distribution of Songs by Mood (2015-2025)',
                    color_discrete_sequence=['#ff6b9d', '#9b7fd4', '#ff8fb3', '#b094d4', '#ff4d7d']
                )
                
                fig6.update_layout(
                    plot_bgcolor='#1a0b2e',
                    paper_bgcolor='#1a0b2e',
                    font_color='white',
                    title_font_size=18,
                    title_font_color='#ff6b9d'
                )
                
                st.plotly_chart(fig6, use_container_width=True)
            
            # Year-wise mood popularity trends
            st.markdown('<div class="section-header">📈 Mood Popularity Over Time</div>', unsafe_allow_html=True)
            
            mood_year_popularity = recent_mood_data.groupby(['year', 'mood_label'])['popularity'].mean().reset_index()
            
            fig7 = px.line(
                mood_year_popularity,
                x='year',
                y='popularity',
                color='mood_label',
                title='Mood Popularity Trends by Year (2015-2025)',
                labels={'year': 'Year', 'popularity': 'Average Popularity', 'mood_label': 'Mood'},
                markers=True,
                color_discrete_sequence=['#ff6b9d', '#9b7fd4', '#ff8fb3', '#b094d4', '#ff4d7d']
            )
            
            fig7.update_layout(
                plot_bgcolor='#1a0b2e',
                paper_bgcolor='#1a0b2e',
                font_color='white',
                title_font_size=18,
                title_font_color='#9b7fd4',
                xaxis=dict(showgrid=True, gridcolor='#2d1b4e'),
                yaxis=dict(showgrid=True, gridcolor='#2d1b4e'),
                legend=dict(
                    bgcolor='#2d1b4e',
                    bordercolor='#9b7fd4',
                    borderwidth=1
                )
            )
            
            st.plotly_chart(fig7, use_container_width=True)
            
            # Most popular song in each mood category
            st.markdown('<div class="section-header">🏆 Top Songs by Mood (2015-2025)</div>', unsafe_allow_html=True)
            
            mood_cols = st.columns(min(len(mood_popularity), 5))
            
            for idx, (_, mood_row) in enumerate(mood_popularity.head(5).iterrows()):
                mood = mood_row['mood_label']
                
                with mood_cols[idx]:
                    st.markdown(f'<div class="category-button">{mood.upper()}</div>', unsafe_allow_html=True)
                    
                    # Get top 3 songs for this mood
                    top_mood_songs = recent_mood_data[recent_mood_data['mood_label'] == mood].nlargest(3, 'popularity')
                    
                    for song_idx, (_, song) in enumerate(top_mood_songs.iterrows()):
                        st.markdown(display_small_card(song), unsafe_allow_html=True)
                        if st.button(f"▶️", key=f"play_mood_analytics_{mood}_{song_idx}", use_container_width=True):
                            st.session_state.selected_song_data = song
                            st.rerun()
        else:
            st.info("No recent year data available for mood analysis")
    else:
        st.warning("Mood or year information not available in the dataset")
    
    # 5. Top Artists with Most Popular Songs
    st.markdown('<div class="section-header">🎤 Artists with Most Popular Songs</div>', unsafe_allow_html=True)
    st.markdown('<p style="color: rgba(255, 255, 255, 0.7); font-size: 14px; margin-bottom: 20px;">Artists who have created the most popular tracks (recent years)</p>', unsafe_allow_html=True)
    
    if 'year' in df.columns:
        recent_artists = df[(df['year'] >= 2015) & (df['year'] <= 2025) & (df['is_popular'])].copy()
    else:
        recent_artists = df[df['is_popular']].copy()
    
    if not recent_artists.empty:
        # Count popular songs per artist
        artist_popularity = recent_artists.groupby('artist_name').agg({
            'track_name': 'count',
            'popularity': 'mean'
        }).reset_index()
        artist_popularity.columns = ['artist_name', 'popular_song_count', 'avg_popularity']
        artist_popularity = artist_popularity.sort_values('popular_song_count', ascending=False).head(15)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Bar chart of top artists
            fig_artist = px.bar(
                artist_popularity.head(10),
                x='artist_name',
                y='popular_song_count',
                title='Top 10 Artists by Popular Song Count',
                labels={'artist_name': 'Artist', 'popular_song_count': 'Number of Popular Songs'},
                color='avg_popularity',
                color_continuous_scale=['#9b7fd4', '#ff6b9d'],
                hover_data={'avg_popularity': ':.2f'}
            )
            
            fig_artist.update_layout(
                plot_bgcolor='#1a0b2e',
                paper_bgcolor='#1a0b2e',
                font_color='white',
                title_font_size=18,
                title_font_color='#9b7fd4',
                xaxis=dict(showgrid=False, tickangle=-45),
                yaxis=dict(showgrid=True, gridcolor='#2d1b4e')
            )
            
            st.plotly_chart(fig_artist, use_container_width=True)
        
        with col2:
            # Display top artist stats
            st.markdown('<div class="section-header" style="font-size: 16px;">🏆 Top Artist</div>', unsafe_allow_html=True)
            
            top_artist = artist_popularity.iloc[0]
            st.markdown(f"""
            <div class="song-card" style="text-align: center; padding: 25px;">
                <h2 style="color: #ff6b9d; margin: 10px 0; font-size: 20px;">{top_artist['artist_name']}</h2>
                <p style="opacity: 0.8; margin: 10px 0;"><strong>{int(top_artist['popular_song_count'])}</strong> Popular Songs</p>
                <p style="opacity: 0.8; margin: 10px 0;">Avg Popularity: <strong>{top_artist['avg_popularity']:.2f}</strong></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show sample songs from top artist
            top_artist_songs = recent_artists[recent_artists['artist_name'] == top_artist['artist_name']].nlargest(3, 'popularity')
            
            st.markdown('<p style="color: #9b7fd4; font-size: 14px; margin-top: 15px; font-weight: 600;">Sample Songs:</p>', unsafe_allow_html=True)
            
            for idx, (_, song) in enumerate(top_artist_songs.iterrows()):
                st.markdown(display_small_card(song), unsafe_allow_html=True)
                if st.button(f"▶️", key=f"play_top_artist_{idx}", use_container_width=True):
                    st.session_state.selected_song_data = song
                    st.rerun()
    
    # 6. Statistics Summary
    st.markdown('<div class="section-header">📊 Popular Songs Statistics</div>', unsafe_allow_html=True)
    
    total_songs = len(df)
    popular_songs = len(df[df['is_popular']])
    popular_percentage = (popular_songs / total_songs) * 100
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="song-card" style="text-align: center;">
            <h2 style="color: #ff6b9d; margin: 0;">{popular_songs:,}</h2>
            <p style="opacity: 0.8; margin: 10px 0 0 0;">Popular Songs</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="song-card" style="text-align: center;">
            <h2 style="color: #9b7fd4; margin: 0;">{popular_percentage:.1f}%</h2>
            <p style="opacity: 0.8; margin: 10px 0 0 0;">Popularity Rate</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        top_lang = lang_popular.iloc[0]['language'] if not lang_popular.empty else 'N/A'
        st.markdown(f"""
        <div class="song-card" style="text-align: center;">
            <h2 style="color: #ff6b9d; margin: 0;">{top_lang}</h2>
            <p style="opacity: 0.8; margin: 10px 0 0 0;">Top Language</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        if 'era' in df.columns and not era_popular.empty:
            top_era = era_popular.loc[era_popular['count'].idxmax(), 'era']
        else:
            top_era = 'N/A'
        st.markdown(f"""
        <div class="song-card" style="text-align: center;">
            <h2 style="color: #9b7fd4; margin: 0;">{top_era}</h2>
            <p style="opacity: 0.8; margin: 10px 0 0 0;">Peak Era</p>
        </div>
        """, unsafe_allow_html=True)


def main():
    # Header with logo and app name
    st.markdown("""
    <div class="main-header">
        <div class="logo">🎵</div>
        <div class="app-name">Reverb</div>
    </div>
    """, unsafe_allow_html=True)
    
    # File path for CSV - USE PREPROCESSED FILE
    CSV_PATH = "spotify_tracks_processed.csv"
    
    # Check if preprocessed file exists
    if not os.path.exists(CSV_PATH):
        st.error("⚠️ **Preprocessed data not found!**")
        st.warning("Please run the preprocessing script first:")
        st.code("python preprocess_data.py", language="bash")
        st.info("""
        **Steps to get started:**
        1. Make sure 'spotify_tracks.csv' is in the same directory
        2. Run: `python preprocess_data.py`
        3. Wait for preprocessing to complete (may take a few minutes)
        4. Run: `streamlit run app.py`
        """)
        return
    
    # Load recommender system
    try:
        recommender = load_recommender(CSV_PATH)
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.info("Please check your CSV file format and try again.")
        return
    
    # Display dataset stats in sidebar
    stats = recommender.get_dataset_stats()
    
    # Music Player (if song is selected)
    if st.session_state.selected_song_data is not None:
        display_music_player(st.session_state.selected_song_data)
    
    # Search bar
    st.markdown('<p style="font-size: 14px; opacity: 0.8; margin-bottom: 5px;">🔍 Search for a song (e.g., "Shape of You")</p>', unsafe_allow_html=True)
    search_query = st.text_input(
        "Search",
        placeholder="Search for a song...",
        label_visibility="collapsed",
        key="search_box"
    )
    
    # Spin Wheel Button (Always visible)
    col_spin, col_space = st.columns([1, 3])
    with col_spin:
        if st.button("🎰 Feeling Lucky? Spin the Wheel!", use_container_width=True):
            st.session_state.show_spin_wheel = True
            st.session_state.spin_result = None
    
    # Show Spin Wheel Modal
    if st.session_state.show_spin_wheel:
        st.markdown("---")
        st.markdown("## 🎰 Fortune Wheel - Pick Your Next Song!")
        
        # Language selection
        all_languages = recommender.get_all_languages()
        selected_languages = st.multiselect(
            "Select languages for the wheel (you can choose multiple):",
            options=all_languages,
            default=[all_languages[0]] if all_languages else []
        )
        
        if selected_languages:
            if st.button("🎯 Start Spinning!", use_container_width=True):
                st.session_state.spin_languages = selected_languages
            
            if 'spin_languages' in st.session_state:
                result = display_spin_wheel(st.session_state.spin_languages, recommender)
                
                # If we got a result from the wheel
                if result:
                    song_data, match_type = recommender.find_song(result)
                    if song_data is not None:
                        st.session_state.spin_result = song_data
                        st.session_state.selected_song_data = song_data
                        st.session_state.show_spin_wheel = False
                        st.rerun()
        else:
            st.warning("Please select at least one language!")
        
        if st.button("❌ Close Wheel", use_container_width=True):
            st.session_state.show_spin_wheel = False
            st.rerun()
        
        st.markdown("---")
    
    # Main tabs - ADDED NEW TAB
    tab1, tab2, tab3 = st.tabs(["🎧 Currently playing", "🎵 You might also like", "📊 Check Out Popular Songs"])
    
    # TAB 1: Currently Playing (ORIGINAL CODE - UNCHANGED)
    with tab1:
        if search_query and search_query.strip():
            with st.spinner("🔍 Searching for song..."):
                song_data, match_type = recommender.find_song(search_query)
            
            if song_data is not None:
                # Display match type message
                if match_type == "fuzzy":
                    st.info(f"🔍 Did you mean: **{song_data['track_name']}**?")
                elif match_type == "exact":
                    st.success(f"✅ Found: **{song_data['track_name']}**")
                
                # Layout: Currently playing on left, recommendations on right
                col1, col2 = st.columns([1, 2], gap="large")
                
                with col1:
                    # Display album image for the searched song
                    search_album_art = song_data.get('artwork_url', '')
                    if not search_album_art or pd.isna(search_album_art) or search_album_art == '':
                        search_album_art = "https://via.placeholder.com/300x300/9b7fd4/ffffff?text=No+Image"
                    
                    st.markdown(f'<img src="{search_album_art}" class="playing-card-image" onerror="this.src=\'https://via.placeholder.com/300x300/9b7fd4/ffffff?text=No+Image\'">', unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class="playing-card">
                        <h3 style="margin-top: 0;">🎵 Track Info</h3>
                        <h2 style="color: #9b7fd4; margin: 15px 0;">{song_data['track_name']}</h2>
                        <p style="font-size: 18px; opacity: 0.9;"><strong>Artist:</strong> {song_data['artist_name']}</p>
                        <p style="opacity: 0.8;"><strong>Album:</strong> {song_data.get('album_name', 'N/A')}</p>
                        <p style="opacity: 0.8;"><strong>Language:</strong> {song_data['language']}</p>
                        <p style="opacity: 0.8;"><strong>Mood:</strong> {song_data.get('mood_label', 'N/A')}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Play button
                    if st.button("▶️ Play This Song", key="play_main_song", use_container_width=True):
                        st.session_state.selected_song_data = song_data
                        st.rerun()
                
                with col2:
                    # Similar songs recommendations
                    st.markdown('<div class="section-header">You might also like</div>', unsafe_allow_html=True)
                    
                    with st.spinner("Finding similar songs..."):
                        similar_songs, _ = recommender.recommend_similar_songs(search_query, 10)
                    
                    if not similar_songs.empty:
                        # Display in 2 columns
                        for i in range(0, len(similar_songs), 2):
                            cols = st.columns(2)
                            for j, col in enumerate(cols):
                                if i + j < len(similar_songs):
                                    song = similar_songs.iloc[i + j]
                                    with col:
                                        st.markdown(
                                            display_song_card(song, song['similarity_score'], "Match"),
                                            unsafe_allow_html=True
                                        )
                                        if st.button(f"▶️ Play", key=f"play_similar_{i}_{j}", use_container_width=True):
                                            st.session_state.selected_song_data = song
                                            st.rerun()
                    else:
                        st.warning("No similar songs found.")
                
                # More from artist section
                st.markdown(f'<div class="section-header">More from {song_data["artist_name"]}</div>', unsafe_allow_html=True)
                
                with st.spinner("Finding more songs by this artist..."):
                    artist_songs, _ = recommender.recommend_by_artist(search_query, 10)
                
                if not artist_songs.empty:
                    # Display in 3 columns
                    cols = st.columns(3)
                    for idx, (_, song) in enumerate(artist_songs.iterrows()):
                        with cols[idx % 3]:
                            st.markdown(
                                display_song_card(song, song['ranking_score'], "Score"),
                                unsafe_allow_html=True
                            )
                            if st.button(f"▶️ Play", key=f"play_artist_{idx}", use_container_width=True):
                                st.session_state.selected_song_data = song
                                st.rerun()
                else:
                    st.info(f"No other songs found by {song_data['artist_name']}")
                
                # ========== REMIX FEATURE ==========
                st.markdown('<div class="section-header">🎧 AI-Generated Remix Playlists</div>', unsafe_allow_html=True)
                st.markdown('<p style="color: rgba(255, 255, 255, 0.7); font-size: 14px; margin-bottom: 20px;">Dynamically generated remixes using Apriori association mining - each remix blends 3-4 tracks with smooth transitions</p>', unsafe_allow_html=True)
                
                with st.spinner("🎨 Generating AI-powered remixes using Apriori algorithm..."):
                    try:
                        remixes = recommender.generate_remix_playlists(song_data['track_name'], num_remixes=7)
                        
                        if remixes and len(remixes) > 0:
                            st.success(f"✨ Generated {len(remixes)} unique remix playlists!")
                            
                            # Display remixes in 2 columns
                            for i in range(0, len(remixes), 2):
                                cols = st.columns(2)
                                for j, col in enumerate(cols):
                                    if i + j < len(remixes):
                                        with col:
                                            display_remix_card(remixes[i + j], i + j, recommender)
                        else:
                            st.info("⚠️ Not enough data to generate remixes for this song. Try searching for a more popular track!")
                    except Exception as e:
                        st.warning(f"⚠️ Could not generate remixes: {str(e)}")
                        st.info("Remix generation requires the mlxtend library. Install it with: pip install mlxtend")
                
            else:
                st.warning("🔍 Song not found. Please try another search or check the spelling.")
                
                # Show some random suggestions
                st.markdown("### Try searching for:")
                random_songs = recommender.df['track_name'].sample(5).tolist()
                for song in random_songs:
                    st.markdown(f"- {song}")
        
        else:
            # Welcome screen
            st.markdown("""
            <div style="text-align: center; padding: 50px; color: white;">
                <h1 style="font-size: 48px; margin-bottom: 20px;">🎵</h1>
                <h2>Welcome to Reverb</h2>
                <p style="font-size: 18px; opacity: 0.8;">Search for a song to discover personalized recommendations</p>
                <p style="font-size: 14px; opacity: 0.6; margin-top: 15px;">✨ Click "▶️ Play This Song" on any track</p>
                <p style="font-size: 14px; opacity: 0.6;">✨ Player appears at top with Play/Pause, Volume & Progress controls</p>
                <p style="font-size: 14px; opacity: 0.6;">✨ Audio plays automatically (if preview available)</p>
                <p style="font-size: 14px; opacity: 0.6;">✨ Click any other song to switch tracks instantly</p>
                <p style="font-size: 14px; opacity: 0.6;">🎧 <strong>NEW:</strong> AI-generated remix playlists using Apriori algorithm!</p>
                <p style="font-size: 16px; opacity: 0.9; margin-top: 20px; color: #ff6b9d;">🎰 <strong>Feeling confused? Click "Spin the Wheel" above!</strong></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show dataset stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"""
                <div class="song-card" style="text-align: center;">
                    <h2 style="color: #9b7fd4; margin: 0;">{stats['total_songs']:,}</h2>
                    <p style="opacity: 0.8; margin: 10px 0 0 0;">Total Songs</p>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="song-card" style="text-align: center;">
                    <h2 style="color: #9b7fd4; margin: 0;">{stats['total_artists']:,}</h2>
                    <p style="opacity: 0.8; margin: 10px 0 0 0;">Artists</p>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class="song-card" style="text-align: center;">
                    <h2 style="color: #9b7fd4; margin: 0;">{stats['languages']}</h2>
                    <p style="opacity: 0.8; margin: 10px 0 0 0;">Languages</p>
                </div>
                """, unsafe_allow_html=True)
            with col4:
                st.markdown(f"""
                <div class="song-card" style="text-align: center;">
                    <h2 style="color: #9b7fd4; margin: 0;">{stats['moods']}</h2>
                    <p style="opacity: 0.8; margin: 10px 0 0 0;">Moods</p>
                </div>
                """, unsafe_allow_html=True)
    
    # TAB 2: You might also like (ORIGINAL CODE - UNCHANGED)
    with tab2:
        # Top Trending section
        st.markdown('<div class="section-header">🔥 Top Trending</div>', unsafe_allow_html=True)
        
        languages = recommender.get_all_languages()[:5]
        
        if languages:
            lang_cols = st.columns(len(languages))
            
            for idx, lang in enumerate(languages):
                with lang_cols[idx]:
                    st.markdown(f'<div class="category-button">TOP {lang.upper()} SONGS</div>', unsafe_allow_html=True)
                    
                    top_songs = recommender.get_top_songs_by_language(lang, 8)
                    
                    for song_idx, (_, song) in enumerate(top_songs.iterrows()):
                        st.markdown(display_small_card(song), unsafe_allow_html=True)
                        if st.button(f"▶️", key=f"play_top_{lang}_{song_idx}", use_container_width=True):
                            st.session_state.selected_song_data = song
                            st.rerun()
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Mood-based playlists section
        st.markdown('<div class="section-header">🎭 Mood based Playlists</div>', unsafe_allow_html=True)
        
        moods = recommender.get_all_moods()
        
        if moods and 'Unknown' in moods:
            moods.remove('Unknown')
        
        # Prioritize specific moods
        priority_moods = ['Silent', 'Chill', 'Workout', 'Sad', 'Happy']
        ordered_moods = [m for m in priority_moods if m in moods]
        ordered_moods.extend([m for m in moods if m not in priority_moods])
        
        if ordered_moods:
            mood_cols = st.columns(min(len(ordered_moods), 5))
            
            for idx, mood in enumerate(ordered_moods[:5]):
                with mood_cols[idx]:
                    st.markdown(f'<div class="category-button">{mood.upper()}</div>', unsafe_allow_html=True)
                    
                    mood_songs = recommender.get_mood_playlist(mood, 8)
                    
                    if not mood_songs.empty:
                        for song_idx, (_, song) in enumerate(mood_songs.iterrows()):
                            st.markdown(display_small_card(song), unsafe_allow_html=True)
                            if st.button(f"▶️", key=f"play_mood_{mood}_{song_idx}", use_container_width=True):
                                st.session_state.selected_song_data = song
                                st.rerun()
                    else:
                        st.markdown('<p style="text-align: center; opacity: 0.5;">No songs</p>', unsafe_allow_html=True)
    
    # TAB 3: Check Out Popular Songs (NEW ANALYTICS TAB)
    with tab3:
        st.markdown("""
        <div style="text-align: center; padding: 30px 0 20px 0; color: white;">
            <h1 style="font-size: 42px; margin-bottom: 15px;">📊 Popular Songs Analytics</h1>
            <p style="font-size: 16px; opacity: 0.8;">Explore trends and statistics about popular music across languages and time periods</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create all analytics visualizations
        create_analytics_visualizations(recommender)


if __name__ == "__main__":
    main()