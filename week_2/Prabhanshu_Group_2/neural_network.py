import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

df = pd.read_csv("mw_pw_profiles.csv")

# Select features and target for prediction
# We'll predict `fantasy_score_total` using runs_scored, balls_faced, fours_scored, sixes_scored, wickets_taken
features = ['runs_scored', 'balls_faced', 'fours_scored', 'sixes_scored', 'wickets_taken']
target = 'fantasy_score_total'

df = df[features + [target]].dropna()

# Normalize features
X = df[features].values
y = df[target].values.reshape(-1, 1)

X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X = (X - X_mean) / X_std

y_mean = np.mean(y)
y_std = np.std(y)
y = (y - y_mean) / y_std

# Set hyperparameters
input_size = X.shape[1]
hidden_size = 10
output_size = 1
learning_rate = 0.01
epochs = 1000

# Initialize weights and biases
np.random.seed(0)
W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))

# Activation functions
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# Loss function
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

losses = []

# Training loop
for epoch in tqdm(range(epochs)):
    # Forward pass
    z1 = np.dot(X, W1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1, W2) + b2
    y_pred = z2

    # Loss calculation
    loss = mse(y, y_pred)
    losses.append(loss)

    # Backward pass
    d_loss = 2 * (y_pred - y) / y.shape[0]

    dW2 = np.dot(a1.T, d_loss)
    db2 = np.sum(d_loss, axis=0, keepdims=True)

    d_hidden = np.dot(d_loss, W2.T) * relu_derivative(z1)
    dW1 = np.dot(X.T, d_hidden)
    db1 = np.sum(d_hidden, axis=0, keepdims=True)

    # Update weights
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    # Logging
    if epoch % 50 == 0:
        print(f"Epoch {epoch}: MSE = {loss:.4f}")

# Plot loss over epochs
plt.plot(losses)
plt.title("Training Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid(True)
plt.show()

# Plot predicted vs actual (denormalized)
y_pred_final = y_pred * y_std + y_mean
y_actual = y * y_std + y_mean

plt.scatter(y_actual, y_pred_final, alpha=0.6)
plt.plot([min(y_actual), max(y_actual)], [min(y_actual), max(y_actual)], color='red')
plt.xlabel("Actual Fantasy Score")
plt.ylabel("Predicted Fantasy Score")
plt.title("Actual vs Predicted Fantasy Score")
plt.grid(True)
plt.show()
