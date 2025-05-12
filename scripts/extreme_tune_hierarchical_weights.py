from __future__ import print_function
from operator import itemgetter
import time
import os

# File paths
DATA_DIR = 'data/'
TEST_FILE = os.path.join(DATA_DIR, 'testItem2.txt')
TRACK_FILE = os.path.join(DATA_DIR, 'trackData2.txt')
GENRE_FILE = os.path.join(DATA_DIR, 'genreData2.txt')
TRAIN_FILE = os.path.join(DATA_DIR, 'trainItem2.txt')

none_value = 0  # Number to replace the none values

# Extreme weight combinations
weight_combinations = [
    # Even more extreme album weights
    {'name': 'album_ultra', 'track': 0.0, 'album': 0.9, 'artist': 0.1, 'genre': 0.0},
    {'name': 'album_max', 'track': 0.0, 'album': 0.95, 'artist': 0.05, 'genre': 0.0},
    # Best so far with small track weight
    {'name': 'super_track_light', 'track': 0.1, 'album': 0.8, 'artist': 0.1, 'genre': 0.0},
    {'name': 'super_track_medium', 'track': 0.15, 'album': 0.75, 'artist': 0.1, 'genre': 0.0},
    # Album + Artist combinations
    {'name': 'album_artist_max', 'track': 0.0, 'album': 0.85, 'artist': 0.15, 'genre': 0.0},
    {'name': 'album_artist_ultra', 'track': 0.0, 'album': 0.9, 'artist': 0.1, 'genre': 0.0},
    # Track + Album combinations with high album weight
    {'name': 'track_album_ultra', 'track': 0.1, 'album': 0.85, 'artist': 0.05, 'genre': 0.0},
    {'name': 'track_album_max', 'track': 0.15, 'album': 0.8, 'artist': 0.05, 'genre': 0.0},
]

# Load user ratings
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
                user_ratings[current_user][item_id] = int(rating)

# Load track to album/artist mapping
print("Loading track mappings...")
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

# Parse test file
print("Loading test data...")
test_tracks = {}
with open(TEST_FILE, 'r') as f:
    current_user = None
    for line in f:
        line = line.strip()
        if '|' in line:
            current_user = line.split('|')[0]
            test_tracks[current_user] = []
        else:
            if line:
                test_tracks[current_user].append(line)

def sort_list(input_list, weights):
    sorted_list = []
    for x in input_list:
        # x = [track_id, track_rating, album_rating, artist_rating, genre1_rating, genre2_rating, ...]
        rat_sum = 0
        counter = 0
        
        # Track rating (if available)
        if x[1] >= 0:
            rat_sum += x[1] * weights['track']
            counter += 1
        
        # Album
        if x[2] >= 0:
            rat_sum += x[2] * weights['album']
            counter += 1
        
        # Artist
        if x[3] >= 0:
            rat_sum += x[3] * weights['artist']
            counter += 1
        
        # Genres
        genre_sum = 0
        genre_count = 0
        for g in x[4:]:
            if g >= 0:
                genre_sum += g
                genre_count += 1
        if genre_count > 0:
            genre_sum = genre_sum / genre_count
            rat_sum += genre_sum * weights['genre']
            counter += 1
        
        # Average
        if counter > 0:
            rat_sum = rat_sum / counter
        
        ratings = int(rat_sum)
        sorted_list.append([x[0], ratings])
    
    # Sort by score ascending
    sorted_list = sorted(sorted_list, key=itemgetter(1))
    pred_dic = {}
    for i, item in enumerate(sorted_list):
        # Top 3 get 0, bottom 3 get 1 (reverse for recommender)
        if i < 3:
            pred_dic[item[0]] = 0
        else:
            pred_dic[item[0]] = 1
    # Return predictions in the order of input_list
    return [pred_dic[x[0]] for x in input_list]

# Try each weight combination
for weights in weight_combinations:
    print(f"\nTrying weights: {weights['name']}")
    print(f"Track: {weights['track']:.2f}, Album: {weights['album']:.2f}, Artist: {weights['artist']:.2f}, Genre: {weights['genre']:.2f}")
    
    start_time = time.time()
    output_file = os.path.join(DATA_DIR, f'output_{weights["name"]}.csv')
    
    with open(output_file, 'w') as predictionFile:
        predictionFile.write('TrackID,Predictor\n')
        for user, tracks in test_tracks.items():
            input_list = []
            for track_id in tracks:
                track_rating = user_ratings.get(user, {}).get(track_id, -1)
                album_id = track_to_album.get(track_id)
                artist_id = track_to_artist.get(track_id)
                genre_ids = track_to_genres.get(track_id, [])
                # Ratings
                album_rating = user_ratings.get(user, {}).get(album_id, none_value) if album_id else none_value
                artist_rating = user_ratings.get(user, {}).get(artist_id, none_value) if artist_id else none_value
                genre_ratings = [user_ratings.get(user, {}).get(g, none_value) for g in genre_ids] if genre_ids else []
                # Compose input: [track_id, track_rating, album_rating, artist_rating, genre1_rating, ...]
                input_list.append([track_id, track_rating, album_rating, artist_rating] + genre_ratings)
            # Get predictions (0/1) for this user's 6 tracks
            preds = sort_list(input_list, weights)
            for track_id, pred in zip(tracks, preds):
                predictionFile.write(f'{user}_{track_id},{pred}\n')
    
    print(f"Finished {weights['name']}, spent {time.time() - start_time:.2f} seconds")
    print(f"Output written to: {output_file}")

print("\nAll weight combinations have been tried!")
print("Please submit each output file to the leaderboard to find the best weights.") 