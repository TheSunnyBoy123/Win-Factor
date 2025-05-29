import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense

df = pd.read_csv('C:/Users/KRITIN KHOWALA/Desktop/Win Factor/mw_pw_profiles.csv')

class LinearRegressionCustom:
    def __init__(self, learning_rate=0.001, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            y_pred = np.dot(X, self.weights) + self.bias
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            loss = np.mean((y_pred - y) ** 2)
            self.loss_history.append(loss)

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

X_batting = df[['runs_scored', 'balls_faced', 'fours_scored', 'sixes_scored', 'dot_balls_as_batsman']]
y_batting = df['fantasy_score_batting']
X_bowling = df[['catches_taken', 'run_out_direct', 'run_out_throw', 'stumpings_done', 'balls_bowled', 'runs_conceded', 'wickets_taken', 'bowled_done', 'lbw_done', 'maidens', 'dot_balls_as_bowler']]
y_bowling = df['fantasy_score_bowling']
X_total = df[['catches_taken', 'run_out_direct', 'run_out_throw', 'stumpings_done', 'balls_bowled', 'runs_conceded', 'wickets_taken', 'bowled_done', 'lbw_done', 'maidens', 'dot_balls_as_bowler', 'runs_scored', 'balls_faced', 'fours_scored', 'sixes_scored', 'dot_balls_as_batsman']]
y_total = df['fantasy_score_total']


def create_model(input_dim):
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(8, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X_batting, y_batting, test_size=0.2, random_state=42)
scaler_batting = StandardScaler()
X_train_b_scaled = scaler_batting.fit_transform(X_train_b)
X_test_b_scaled = scaler_batting.transform(X_test_b)
model_batting = create_model(input_dim=X_train_b_scaled.shape[1])
model_batting.fit(X_train_b_scaled, y_train_b, epochs=50, batch_size=32, verbose=0)
y_pred_b = model_batting.predict(X_test_b_scaled).flatten()
mse_b = mean_squared_error(y_test_b, y_pred_b)
r2_b = r2_score(y_test_b, y_pred_b)
print(f"Neural Net Model - Batting: MSE={mse_b:.4f}, R²={r2_b:.4f}")

X_train_bw, X_test_bw, y_train_bw, y_test_bw = train_test_split(X_bowling, y_bowling, test_size=0.2, random_state=42)
scaler_bowling = StandardScaler()
X_train_bw_scaled = scaler_bowling.fit_transform(X_train_bw)
X_test_bw_scaled = scaler_bowling.transform(X_test_bw)
model_bowling = create_model(input_dim=X_train_bw_scaled.shape[1])
model_bowling.fit(X_train_bw_scaled, y_train_bw, epochs=50, batch_size=32, verbose=0)
y_pred_bw = model_bowling.predict(X_test_bw_scaled).flatten()
mse_bw = mean_squared_error(y_test_bw, y_pred_bw)
r2_bw = r2_score(y_test_bw, y_pred_bw)
print(f"Neural Net Model - Bowling: MSE={mse_bw:.4f}, R²={r2_bw:.4f}")

X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(X_total, y_total, test_size=0.2, random_state=42)
scaler_total = StandardScaler()
X_train_t_scaled = scaler_total.fit_transform(X_train_t)
X_test_t_scaled = scaler_total.transform(X_test_t)
model_total = create_model(input_dim=X_train_t_scaled.shape[1])
model_total.fit(X_train_t_scaled, y_train_t, epochs=50, batch_size=32, verbose=0)
y_pred_t = model_total.predict(X_test_t_scaled).flatten()
mse_t = mean_squared_error(y_test_t, y_pred_t)
r2_t = r2_score(y_test_t, y_pred_t)
print(f"Neural Net Model - Total: MSE={mse_t:.4f}, R²={r2_t:.4f}")
