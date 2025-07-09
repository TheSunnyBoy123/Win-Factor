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

X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X_batting, y_batting, test_size=0.2, random_state=42)
scaler_b = StandardScaler()
X_train_b_scaled = scaler_b.fit_transform(X_train_b)
X_test_b_scaled = scaler_b.transform(X_test_b)
model_b = LinearRegressionCustom(learning_rate=0.001, n_iterations=10000)
model_b.fit(X_train_b_scaled, y_train_b)
y_pred_b = model_b.predict(X_test_b_scaled)
mse_b = mean_squared_error(y_test_b, y_pred_b)
r2_b = r2_score(y_test_b, y_pred_b)
print(f"Linear Model - Batting: MSE={mse_b:.4f}, R²={r2_b:.4f}")

X_train_bw, X_test_bw, y_train_bw, y_test_bw = train_test_split(X_bowling, y_bowling, test_size=0.2, random_state=42)
scaler_bw = StandardScaler()
X_train_bw_scaled = scaler_bw.fit_transform(X_train_bw)
X_test_bw_scaled = scaler_bw.transform(X_test_bw)
model_bw = LinearRegressionCustom(learning_rate=0.001, n_iterations=10000)
model_bw.fit(X_train_bw_scaled, y_train_bw)
y_pred_bw = model_bw.predict(X_test_bw_scaled)
mse_bw = mean_squared_error(y_test_bw, y_pred_bw)
r2_bw = r2_score(y_test_bw, y_pred_bw)
print(f"Linear Model - Bowling: MSE={mse_bw:.4f}, R²={r2_bw:.4f}")

X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(X_total, y_total, test_size=0.2, random_state=42)
scaler_t = StandardScaler()
X_train_t_scaled = scaler_t.fit_transform(X_train_t)
X_test_t_scaled = scaler_t.transform(X_test_t)
model_t = LinearRegressionCustom(learning_rate=0.001, n_iterations=10000)
model_t.fit(X_train_t_scaled, y_train_t)
y_pred_t = model_t.predict(X_test_t_scaled)
mse_t = mean_squared_error(y_test_t, y_pred_t)
r2_t = r2_score(y_test_t, y_pred_t)
print(f"Linear Model - Total: MSE={mse_t:.4f}, R²={r2_t:.4f}")