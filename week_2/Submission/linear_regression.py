import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('C:/Users/chhon/Downloads/mw_pw_profiles.csv')
numeric_df = df.select_dtypes(include=[np.number])
class LinearRegressionModel:
    def __init__(self):
        self.m = None
        self.c = None
    def fit(self, X, y):
        X_mean = np.mean(X)
        y_mean = np.mean(y)
        self.m = np.sum((X - X_mean) * (y - y_mean)) / np.sum((X - X_mean) ** 2)
        self.c = y_mean - self.m * X_mean
    def predict(self, X):
        return self.m * X + self.c
    def r2_score(self, X, y):
        y_pred = self.predict(X)
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)
        return 1 - (ss_residual / ss_total)
    def rmse(self, X, y):
        y_pred = self.predict(X)
        return np.sqrt(np.mean((y - y_pred) ** 2))

split = int(len(numeric_df) * 0.8)
X = numeric_df['runs_scored'].values
y = numeric_df['fantasy_score_batting'].values
X_train = X[:split]
y_train = y[:split]
X_test = X[split:]
y_test = y[split:]
model = LinearRegressionModel()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Slope (m):", model.m)
print("Intercept (c):", model.c)
print("R2 Score:", model.r2_score(X_test, y_test))
print("RMSE:", model.rmse(X_test, y_test))
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', label='Predicted')
plt.xlabel('Runs Scored')
plt.ylabel('Fantasy Score Batting')
plt.title('Linear Regression: Runs Scored vs Fantasy Score Batting')
plt.legend()
plt.show()
