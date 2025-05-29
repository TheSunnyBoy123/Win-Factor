import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
df = pd.read_csv("mw_pw_profiles.csv")

#print(df.columns)
X = df['fantasy_score_batting'].values
Y = df['runs_scored'].values
n=len(X)
M=0
C=0
lr=0.1
iterations=10000
error=[]

#Stanadardizing
X_mean = np.mean(X)
X_std = np.std(X)
X = (X-X_mean)/X_std

for i in tqdm(range(iterations)):

 Y_pred = M*X + C
 error = Y-Y_pred
 grad_M = -(2/n)*np.sum(X*error)
 grad_C = -(2/n)*np.sum(error)
 M=M-lr*grad_M
 C=C-lr*grad_C

Y_pred= M*X +C
X_actual= df['fantasy_score_batting'].values

#To test accuracy
r2= 1-(np.sum((Y - Y_pred)**2)/np.sum((Y-np.mean(Y))**2))
mse= np.mean((Y - Y_pred)**2)


print(f"Slope (m): {M:.2f}")
print(f"Intercept (c): {C:.2f}")
print(f"R2 Score: {r2:.4f}")
print(f"MSE: {mse:.4f}")

# Plot
plt.figure(figsize=(10,7),facecolor='lightblue')
plt.scatter(X_actual, Y, color='greenyellow', edgecolors='k',alpha=0.8)
plt.plot(X_actual, Y_pred, color='firebrick', linewidth=4)
plt.title('Linear regression: Fantasy Score Batting vs Runs Scored',fontsize=20, fontweight='bold', color='indigo')
plt.xlabel('Fantasy Score Batting',fontweight='bold')
plt.ylabel('Runs Scored',fontweight='bold')
plt.grid(True,color='blue')
plt.tight_layout()
plt.show()