import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

df = pd.read_csv("C:\\Users\\Administrator\\Downloads\\mw_pw_profiles.txt", low_memory=False)

class FantasyScoreNN(nn.Module):
    def __init__(self, input_size):
        super(FantasyScoreNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)

def prepare_data(df, feature_cols, target_col, test_size=0.2):
    X = df[feature_cols].values
    y = df[[target_col]].values

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=test_size, random_state=42)

    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
        scaler_X,
        scaler_y
    )

def train_model(model, X_train, y_train, X_test, y_test, epochs=100, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test)
                test_loss = criterion(test_outputs, y_test)
            print(f"Epoch {epoch+1}, Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        y_pred = model(X_test).numpy()
        y_true = y_test.numpy()

    return model, y_pred, y_true

batting_features = ['runs_scored','player_out','balls_faced','fours_scored','sixes_scored','dot_balls_as_batsman','order_seen']
bowling_features = ['catches_taken','run_out_direct','run_out_throw','stumpings_done','balls_bowled','runs_conceded','wickets_taken','bowled_done','lbw_done','maidens','dot_balls_as_bowler']
total_features = batting_features + bowling_features

X_train, X_test, y_train, y_test, scaler_X_bat, scaler_y_bat = prepare_data(df, batting_features, 'fantasy_score_batting')
bat_model = FantasyScoreNN(len(batting_features))
bat_model, bat_pred_scaled, bat_true_scaled = train_model(bat_model, X_train, y_train, X_test, y_test)
bat_pred = scaler_y_bat.inverse_transform(bat_pred_scaled)
bat_true = scaler_y_bat.inverse_transform(bat_true_scaled)
print(f"R² Score (Batting): {r2_score(bat_true, bat_pred):.4f}\n")

X_train, X_test, y_train, y_test, scaler_X_bowl, scaler_y_bowl = prepare_data(df, bowling_features, 'fantasy_score_bowling')
bowl_model = FantasyScoreNN(len(bowling_features))
bowl_model, bowl_pred_scaled, bowl_true_scaled = train_model(bowl_model, X_train, y_train, X_test, y_test)
bowl_pred = scaler_y_bowl.inverse_transform(bowl_pred_scaled)
bowl_true = scaler_y_bowl.inverse_transform(bowl_true_scaled)
print(f"R² Score (Bowling): {r2_score(bowl_true, bowl_pred):.4f}\n")

X_train, X_test, y_train, y_test, scaler_X_total, scaler_y_total = prepare_data(df, total_features, 'fantasy_score_total')
total_model = FantasyScoreNN(len(total_features))
total_model, total_pred_scaled, total_true_scaled = train_model(total_model, X_train, y_train, X_test, y_test)
total_pred = scaler_y_total.inverse_transform(total_pred_scaled)
total_true = scaler_y_total.inverse_transform(total_true_scaled)
print(f"R² Score (Total): {r2_score(total_true, total_pred):.4f}")
