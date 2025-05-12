import csv
import os
from collections import Counter, defaultdict

DATA_DIR = 'data/'
output_files = [
    'output_album_super.csv',
    'output_album_heaviest.csv',
    'output_album_heavy.csv',
    'output_track_album_max.csv',
    'output_album_max.csv',
]
output_files = [os.path.join(DATA_DIR, f) for f in output_files]

# Read all predictions into a dict: TrackID -> [pred1, pred2, ...]
predictions = defaultdict(list)
header = None
for file in output_files:
    with open(file, 'r') as f:
        reader = csv.reader(f)
        file_header = next(reader)
        if header is None:
            header = file_header
        for row in reader:
            track_id, pred = row[0], int(row[1])
            predictions[track_id].append(pred)

# Majority voting
def majority_vote(preds):
    count = Counter(preds)
    if count[0] > count[1]:
        return 0
    else:
        return 1  # If tie, or more 1s, return 1

ensemble_preds = {tid: majority_vote(preds) for tid, preds in predictions.items()}

# Write ensemble output
ensemble_file = os.path.join(DATA_DIR, 'output_ensemble.csv')
with open(ensemble_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for tid in sorted(ensemble_preds):
        writer.writerow([tid, ensemble_preds[tid]])

print(f"Ensemble predictions written to {ensemble_file}") 