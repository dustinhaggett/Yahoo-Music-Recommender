# Yahoo Music Recommender

This repository contains code and scripts for the Yahoo Music Recommender Challenge, leveraging hierarchical features (track, album, artist, genre) and a variety of rule-based and ensemble methods.

## Project Structure

```
scripts/
  generate_hierarchical_features.py         # Generate hierarchical features for each user-track pair
  predict_by_album_artist_sum.py            # Simple baseline: sum album and artist ratings
  predict_by_weighted_hierarchical_sum.py   # Weighted sum of album, artist, genre ratings
  grid_search_hierarchical_weights.py       # Grid search for best hierarchical weights
  fine_tune_hierarchical_weights.py         # Fine-tune weights around best values
  extreme_tune_hierarchical_weights.py      # Extreme weight combinations
  final_tune_hierarchical_weights.py        # Final round of weight tuning
  ensemble_majority_vote_predictions.py     # Ensemble predictions from top models
  train_ml_classifier_on_features.py        # (Optional) ML classifier on engineered features
data/
  trainItem2.txt
  testItem2.txt
  trackData2.txt
  genreData2.txt
  ... (other data files)
```

## Workflow

1. **Feature Engineering**
   - Run `generate_hierarchical_features.py` to create a feature table for each user-track pair in the test set.

2. **Rule-Based Prediction**
   - Use `predict_by_album_artist_sum.py` for a simple baseline (sum of album and artist ratings).
   - Use `predict_by_weighted_hierarchical_sum.py` or the weight tuning scripts to try different weighted combinations.

3. **Weight Tuning**
   - Use the grid/fine/extreme/final tuning scripts to search for the best weights for album, artist, and genre ratings.

4. **Ensembling**
   - Use `ensemble_majority_vote_predictions.py` to combine predictions from the top-performing models.

5. **(Optional) Machine Learning**
   - Use `train_ml_classifier_on_features.py` to train ML models (e.g., logistic regression, random forest) on the engineered features.

## Usage

1. **Set up your environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Run feature generation:**
   ```bash
   python scripts/generate_hierarchical_features.py
   ```

3. **Run a prediction script:**
   ```bash
   python scripts/predict_by_album_artist_sum.py
   # or
   python scripts/predict_by_weighted_hierarchical_sum.py
   ```

4. **Tune weights (optional):**
   ```bash
   python scripts/grid_search_hierarchical_weights.py
   ```

5. **Ensemble predictions:**
   ```bash
   python scripts/ensemble_majority_vote_predictions.py
   ```

6. **Submit the resulting CSV file (e.g., `data/output_ensemble.csv`) to the leaderboard.**

## Results

- The best individual and ensemble models achieved a score of **0.846** on the leaderboard.
- Album and artist ratings were the most predictive features.
- Ensembling provided robust, consistent results.

## Next Steps

- Explore collaborative filtering or neural models for further improvement.
- Engineer new features (user/item statistics, popularity, etc.).
- Try stacking or weighted ensembling for more diversity.

## License

MIT License

---

