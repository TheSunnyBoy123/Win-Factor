#I am making a model which uses linear regression which takes selected inputs to train and asks player_id to give the predicted fantasy point

import numpy as np
import pandas as pd

df = pd.read_csv('mw_pw_profiles.txt')

features = [
    'runs_scored', 'balls_faced', 'fours_scored', 'sixes_scored',
    'catches_taken', 'run_out_direct', 'run_out_throw', 'stumpings_done',
    'balls_bowled', 'runs_conceded', 'wickets_taken', 'maidens', 'dot_balls_as_bowler'
]

df = df.fillna(0)
X = df[features].values
Y = df['fantasy_score_total'].values

# Normalize features
mean = X.mean(axis=0)
std = X.std(axis=0)
X_norm = (X - mean) / std

X_train = X_norm
y_train = Y

weights = np.zeros(len(features))
bias = 0
learning_rate = 0.01
iterations = 1000
n_samples = X_train.shape[0]

for _ in range(iterations):
    y_pred = np.dot(X_train, weights) + bias
    dw = (2 / n_samples) * np.dot(X_train.T, (y_pred - y_train))
    db = (2 / n_samples) * np.sum(y_pred - y_train)

    weights -= learning_rate * dw
    bias -= learning_rate * db

print("Model training complete!")

while True:
    player_id = input("\nEnter player id or type 'exit' to quit: ")

    if player_id.lower() == 'exit':
        break

    # Filter data for the player
    player_data = df[df['player_id'] == player_id]

    if player_data.empty:
        print(f"No data found for player: {player_id}")
        continue

    # Calculate average historical stats for the player
    player_avg = player_data[features].mean().values

    # Normalize using training mean/std
    player_avg_norm = (player_avg - mean) / std

    # Predict fantasy score
    predicted_score = np.dot(player_avg_norm, weights) + bias

    print(f"Predicted Fantasy Score for {player_id}: {predicted_score:.2f}")
