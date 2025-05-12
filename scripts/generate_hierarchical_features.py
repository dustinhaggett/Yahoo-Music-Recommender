from __future__ import print_function
import time
import os

# File paths
DATA_DIR = 'data/'
TEST_FILE = os.path.join(DATA_DIR, 'testItem2.txt')
TRAIN_FILE = os.path.join(DATA_DIR, 'trainItem2.txt')
TRACK_FILE = os.path.join(DATA_DIR, 'trackData2.txt')
GENRE_FILE = os.path.join(DATA_DIR, 'genreData2.txt')
OUTPUT_FILE = os.path.join(DATA_DIR, 'test_hie_score.txt')

# Load track to album/artist mapping
track_to_album = {}
track_to_artist = {}
with open(TRACK_FILE, 'r') as f:
    for line in f:
        parts = line.strip().split('|')
        if len(parts) >= 3:
            track_id = parts[0]
            album_id = parts[1]
            artist_id = parts[2]
            track_to_album[track_id] = album_id
            track_to_artist[track_id] = artist_id

# Load track to genre mapping
track_to_genres = {}
with open(GENRE_FILE, 'r') as f:
    for line in f:
        parts = line.strip().split('|')
        if len(parts) >= 2:
            track_id = parts[0]
            genre_id = parts[1]
            if track_id not in track_to_genres:
                track_to_genres[track_id] = []
            track_to_genres[track_id].append(genre_id)

# Load user ratings from training file
print("Loading user ratings...")
user_ratings = {}
with open(TRAIN_FILE, 'r') as f:
    current_user = None
    for line in f:
        line = line.strip()
        if '|' in line:
            current_user = line.split('|')[0]
            user_ratings[current_user] = {}
        else:
            if line:
                item_id, rating = line.split()
                user_ratings[current_user][item_id] = rating

# Process test file and write features
print("Generating hierarchical features for test set...")
start_time = time.time()
with open(TEST_FILE, 'r') as testData, open(OUTPUT_FILE, 'w') as outFile:
    current_user = None
    for line in testData:
        line = line.strip()
        if '|' in line:
            current_user = line.split('|')[0]
        else:
            track_id = line
            album_id = track_to_album.get(track_id, 'None')
            artist_id = track_to_artist.get(track_id, 'None')
            genre_ids = track_to_genres.get(track_id, [])
            # Get ratings
            track_rating = user_ratings.get(current_user, {}).get(track_id, 'None')
            album_rating = user_ratings.get(current_user, {}).get(album_id, 'None') if album_id != 'None' else 'None'
            artist_rating = user_ratings.get(current_user, {}).get(artist_id, 'None') if artist_id != 'None' else 'None'
            genre_ratings = [user_ratings.get(current_user, {}).get(g, 'None') for g in genre_ids] if genre_ids else []
            # Write: user|track|track_rating|album_rating|artist_rating|genre1_rating|...
            outFile.write(f"{current_user}|{track_id}|{track_rating}|{album_rating}|{artist_rating}")
            for gr in genre_ratings:
                outFile.write(f"|{gr}")
            outFile.write("\n")
print(f"Finished, spent {time.time() - start_time:.2f} s") 