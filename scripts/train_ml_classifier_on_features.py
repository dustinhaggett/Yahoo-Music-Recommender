import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import os

DATA_DIR = 'data/'
FEATURES_FILE = os.path.join(DATA_DIR, 'output1.txt')  # userID|trackID|album_score|artist_score
GROUND_TRUTH_FILE = os.path.join(DATA_DIR, 'test2_new.txt')  # userID|trackID|ground_truth
OUTPUT_FILE = os.path.join(DATA_DIR, 'ml_predictions.csv')

# Load features
df_features = pd.read_csv(FEATURES_FILE, sep='|', names=['userID', 'trackID', 'album_score', 'artist_score'])

# Load ground truth
df_truth = pd.read_csv(GROUND_TRUTH_FILE, sep='|', names=['userID', 'trackID', 'ground_truth'])

# Merge for training
df_train = pd.merge(df_truth, df_features, on=['userID', 'trackID'], how='inner').fillna(0)

# Features and label
X = df_train[['album_score', 'artist_score']]
y = df_train['ground_truth']

# Split for validation
df_Xtrain, df_Xval, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# Logistic Regression
lr = LogisticRegression(max_iter=200)
lr.fit(df_Xtrain, y_train)
y_pred_lr = lr.predict(df_Xval)
auc_lr = roc_auc_score(y_val, y_pred_lr)
acc_lr = accuracy_score(y_val, y_pred_lr)
print(f'Logistic Regression: AUC={auc_lr:.3f}, Accuracy={acc_lr:.3f}')

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(df_Xtrain, y_train)
y_pred_rf = rf.predict(df_Xval)
auc_rf = roc_auc_score(y_val, y_pred_rf)
acc_rf = accuracy_score(y_val, y_pred_rf)
print(f'Random Forest: AUC={auc_rf:.3f}, Accuracy={acc_rf:.3f}')

# Predict on all 120,000 for submission (no ground truth)
X_test = df_features[['album_score', 'artist_score']]
preds_lr = lr.predict(X_test)
preds_rf = rf.predict(X_test)

# Majority vote ensemble
preds_ensemble = (preds_lr + preds_rf) >= 1  # at least one model says 1
preds_ensemble = preds_ensemble.astype(int)

# Prepare output
output_df = df_features[['userID', 'trackID']].copy()
output_df['Predictor'] = preds_ensemble
output_df['TrackID'] = output_df['userID'].astype(str) + '_' + output_df['trackID'].astype(str)
output_df = output_df[['TrackID', 'Predictor']]
output_df.to_csv(OUTPUT_FILE, index=False)
print(f'Predictions written to {OUTPUT_FILE}') 