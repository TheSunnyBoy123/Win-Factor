import numpy as np
import pandas as pd

# Load and clean data
df = pd.read_csv('mw_pw_profiles.csv')
df = df[['runs', 'balls_faced']].dropna()

X = df['balls_faced'].values.reshape(-1, 1)
y = df['runs'].values.reshape(-1, 1)

# Add bias term
X_b = np.c_[np.ones((X.shape[0], 1)), X]

# Linear Regression using Normal Equation
theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

# Predict
y_pred = X_b @ theta

# Print learned parameters and MSE
print("Theta (weights):", theta.ravel())
print("MSE:", np.mean((y - y_pred)**2))
