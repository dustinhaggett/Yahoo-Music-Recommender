#!/usr/bin/env python

import csv
import logging
import time
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# File paths
dataDir = 'data/'
train_file = dataDir + 'trainItem2.txt'
test_file = dataDir + 'testItem2.txt'
output_file = dataDir + 'output1.csv'
track_file = dataDir + 'trackData2.txt'
album_file = dataDir + 'albumData2.txt'
artist_file = dataDir + 'artistData2.txt'
genre_file = dataDir + 'genreData2.txt'

start_time = time.time()

# 1. Load user ratings
logging.info("Loading user ratings...")
user_ratings = {}
with open(train_file, 'r') as f:
    current_user = None
    for line in f:
        line = line.strip()
        if '|' in line:
            current_user = line.split('|')[0]
            user_ratings[current_user] = {}
        else:
            if line:  # Skip empty lines
                item_id, rating = line.split()
                user_ratings[current_user][item_id] = rating

logging.info(f"Loaded ratings for {len(user_ratings)} users")

# 2. Load hierarchy data
logging.info("Loading hierarchy data...")
track_to_album = {}
track_to_artist = {}
track_to_genre = {}

# Load track data
with open(track_file, 'r') as f:
    for line in f:
        parts = line.strip().split('|')
        if len(parts) >= 3:
            track_id = parts[0]
            album_id = parts[1]
            artist_id = parts[2]
            track_to_album[track_id] = album_id
            track_to_artist[track_id] = artist_id

# Load genre data
with open(genre_file, 'r') as f:
    for line in f:
        parts = line.strip().split('|')
        if len(parts) >= 2:
            track_id = parts[0]
            genre_id = parts[1]
            track_to_genre[track_id] = genre_id

# 3. Process test file and generate predictions
logging.info("Processing test file and generating predictions...")
results = []
current_user = None
user_tracks = []

with open(test_file, 'r') as f:
    for line in f:
        line = line.strip()
        if '|' in line:
            if user_tracks:
                # Process previous user's tracks
                user_tracks.sort(key=lambda x: x[1], reverse=True)
                for j, (track, _) in enumerate(user_tracks):
                    pred = 1 if j < 3 else 0
                    results.append((f"{current_user}_{track}", pred))
            
            current_user = line.split('|')[0]
            user_tracks = []
        else:
            if line:  # Skip empty lines
                track_id = line
                
                # Get hierarchy information
                album_id = track_to_album.get(track_id)
                artist_id = track_to_artist.get(track_id)
                genre_id = track_to_genre.get(track_id)
                
                # Get scores from user ratings
                album_score = user_ratings.get(current_user, {}).get(album_id, 'None') if album_id else 'None'
                artist_score = user_ratings.get(current_user, {}).get(artist_id, 'None') if artist_id else 'None'
                genre_score = user_ratings.get(current_user, {}).get(genre_id, 'None') if genre_id else 'None'
                
                # Calculate score based on hierarchy
                score = 0
                
                # Case 1: Both album and artist are rated 1
                if album_score == '1' and artist_score == '1':
                    score = 1
                
                # Case 2: Either album or artist is rated 1
                elif album_score == '1' or artist_score == '1':
                    score = 1
                
                # Case 3: Neither album nor artist is rated 1, but genre is rated 1
                elif album_score != '1' and artist_score != '1' and genre_score == '1':
                    score = 1
                
                user_tracks.append((track_id, score))

# Process last user's tracks
if user_tracks:
    user_tracks.sort(key=lambda x: x[1], reverse=True)
    for j, (track, _) in enumerate(user_tracks):
        pred = 1 if j < 3 else 0
        results.append((f"{current_user}_{track}", pred))

# 4. Write output in sample submission format
logging.info("Writing predictions to file...")
with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['TrackID', 'Predictor'])
    for trackid, pred in results:
        writer.writerow([trackid, pred])

logging.info(f"Done! Time spent: {time.time() - start_time:.2f} seconds")  
