
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load and prepare data
df = pd.read_csv('mw_pw_profiles.csv')
df = df[['runs', 'balls_faced', 'strike_rate']].dropna()

X = df[['balls_faced', 'strike_rate']].values
y = df[['runs']].values

# Normalize
X = (X - X.mean(axis=0)) / X.std(axis=0)
y = y / y.max()

# Parameters
input_size = X.shape[1]
hidden_size = 5
output_size = 1
lr = 0.01
epochs = 1000

# Initialize weights
W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))

# Activation functions
def relu(z):
    return np.maximum(0, z)

def relu_deriv(z):
    return z > 0

# Training loop
for epoch in range(epochs):
    z1 = X @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    y_pred = z2

    loss = np.mean((y - y_pred) ** 2)

    dloss = 2 * (y_pred - y) / y.shape[0]
    dW2 = a1.T @ dloss
    db2 = np.sum(dloss, axis=0, keepdims=True)

    da1 = dloss @ W2.T
    dz1 = da1 * relu_deriv(z1)
    dW1 = X.T @ dz1
    db1 = np.sum(dz1, axis=0, keepdims=True)

    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.4f}")

# Evaluate
z1 = X @ W1 + b1
a1 = relu(z1)
y_pred = a1 @ W2 + b2

# Rescale prediction
y_pred_rescaled = y_pred * df['runs'].max()
y_true = y * df['runs'].max()

plt.scatter(y_true, y_pred_rescaled, alpha=0.5)
plt.xlabel("Actual Runs")
plt.ylabel("Predicted Runs")
plt.title("Neural Network Predictions")
plt.plot([0, y_true.max()], [0, y_true.max()], 'r--')
plt.grid(True)
plt.show()
