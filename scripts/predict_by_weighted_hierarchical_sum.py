from __future__ import print_function
from operator import itemgetter
import time
import os

# File paths
DATA_DIR = 'data/'
RESULT_FILE = os.path.join(DATA_DIR, 'output_album_artist_genre_weights.csv')
TEST_FILE = os.path.join(DATA_DIR, 'testItem2.txt')
TRACK_FILE = os.path.join(DATA_DIR, 'trackData2.txt')
GENRE_FILE = os.path.join(DATA_DIR, 'genreData2.txt')
TRAIN_FILE = os.path.join(DATA_DIR, 'trainItem2.txt')

none_value = 0  # Number to replace the none values
album_weight = 0.4
artist_weight = 0.3
genre_weight = 0.3

# Load user ratings
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

# Load track to genre mapping (can be multiple genres per track)
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

# Parse test file into a dict: user -> list of 6 track_ids
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

# Main weighted prediction logic with track-level override
def sort_list(input_list):
    sorted_list = []
    for x in input_list:
        # x = [track_id, track_rating, album_rating, artist_rating, genre1_rating, genre2_rating, ...]
        # If user has rated the track, use that as the score
        if x[1] >= 0:
            rat_sum = x[1]  # Use track rating directly
        else:
            rat_sum = 0
            counter = 0
            genre_sum = 0
            genre_count = 0
            # Album
            if x[2] >= 0:
                rat_sum += x[2] * album_weight
                counter += 1
            # Artist
            if x[3] >= 0:
                rat_sum += x[3] * artist_weight
                counter += 1
            # Genres
            for g in x[4:]:
                if g >= 0:
                    genre_sum += g
                    genre_count += 1
            if genre_count > 0:
                genre_sum = genre_sum / genre_count
                rat_sum += genre_sum * genre_weight
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

start_time = time.time()
with open(RESULT_FILE, 'w') as predictionFile:
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
        preds = sort_list(input_list)
        for track_id, pred in zip(tracks, preds):
            predictionFile.write(f'{user}_{track_id},{pred}\n')
print("Finished, spend %.2f s" % (time.time() - start_time)) 