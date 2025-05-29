import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load and clean data
df = pd.read_csv('mw_pw_profiles.csv')
df = df[['runs', 'balls_faced']].dropna()

X = df['balls_faced'].values.reshape(-1, 1)
y = df['runs'].values.reshape(-1, 1)

# Polynomial feature generator
def polynomial_features(X, degree):
    return np.hstack([X ** i for i in range(degree + 1)])

# Generate polynomial features
degree = 3
X_poly = polynomial_features(X, degree)

# Normal Equation for polynomial regression
theta = np.linalg.inv(X_poly.T @ X_poly) @ X_poly.T @ y

y_pred = X_poly @ theta

# Plot actual vs predicted
plt.scatter(X, y, color='blue', label='Actual')
plt.scatter(X, y_pred, color='red', s=10, label=f'Predicted (deg {degree})')
plt.xlabel("Balls Faced")
plt.ylabel("Runs")
plt.legend()
plt.title("Polynomial Regression")
plt.show()

print("Theta (weights):", theta.ravel())
print("MSE:", np.mean((y - y_pred)**2))
