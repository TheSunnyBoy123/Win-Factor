#Polynomial regression to predict Runs scored from number of boundaries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv('mw_pw_profiles.csv')
df['bdry'] = df['fours_scored'] + df['sixes_scored'] + df['balls_faced']*0.1
df = df[(df['bdry'] <= 110) & (df['runs_scored'] <= 400)] # removes outliers

X = df['bdry'].values
Y = df['runs_scored'].values
#print(df.head())
n=len(X)
X1=X
X2=X**2
X3=X**3

#Standardizing to avoid problems with gradient descent
X1_mean=np.mean(X1)
X2_mean=np.mean(X2)
X3_mean=np.mean(X3)

X1_std=np.std(X1)
X2_std=np.std(X2)
X3_std=np.std(X3)

X1 = (X1 - X1_mean)/ X1_std
X2 = (X2 - X2_mean)/X2_std
X3 = (X3 - X3_mean)/X3_std

a1 = 0
a2 = 0
a3 = 0
C = 0
lr = 0.219  # Hit and trial
iterations = 100000  # Hit and trial

# Regression using gradient descent
for i in range(iterations):
    Y_pred = a1 * X1 + a2 * X2 + a3 * X3 + C
    error = Y - Y_pred

    grad_a1 = -(2 / n) * np.sum(X1 * error)
    grad_a2 = -(2 / n) * np.sum(X2 * error)
    grad_a3 = -(2 / n) * np.sum(X3 * error)
    grad_C = -(2 / n) * np.sum(error)

    a1 -= lr * grad_a1
    a2 -= lr * grad_a2
    a3 -= lr * grad_a3
    C -= lr * grad_C
# Model accuracy tests
r2 = 1 - (np.sum((Y - Y_pred) ** 2) / np.sum((Y - np.mean(Y)) ** 2))
mse = np.mean((Y - Y_pred) ** 2)

print(f"X-Intercept: {a1:.2f}")
print(f"X^2-Intercept: {a2:.2f}")
print(f"X^3-Intercept: {a3:.2f}")
print(f"Intercept of C: {C:.2f}")
print(f"R2 Score: {r2:.4f}")
print(f"MSE: {mse:.2f}")

# Standardizing feature again
X_actual = df['bdry'].values
X1_real = (X_actual - X1_mean) / X1_std
X2_real = ((X_actual ** 2) - X2_mean) / X2_std
X3_real = ((X_actual ** 3) - X3_mean) / X3_std
Y_pred_final = a1 * X1_real + a2 * X2_real + a3 * X3_real + C

# multiple polynomial lines were coming hence used this
plot_df = pd.DataFrame({'X': X_actual, 'Y_pred': Y_pred_final})
plot_df = plot_df.sort_values(by='X')

plt.figure(figsize=(8, 6), facecolor='greenyellow')
plt.scatter(X_actual, Y, color='lavenderblush', edgecolors='black', alpha=0.7)
plt.plot(plot_df['X'], plot_df['Y_pred'], color='darkblue', linewidth=4)
plt.title('Boundary Count vs Total Runs(Polynomial regression)', fontsize=19, fontweight='bold', color='indigo')
plt.xlabel('Boundary count', fontweight='bold')
plt.ylabel('Runs Scored', fontweight='bold')
plt.grid(True, color='green')
plt.tight_layout()
plt.show()