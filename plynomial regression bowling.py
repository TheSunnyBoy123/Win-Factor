import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


file_path = "mw_pw_profiles.xlsx" 
columns_needed = [
    'balls_bowled', 'runs_conceded', 'wickets_taken',
    'bowled_done', 'lbw_done', 'maidens',
    'dot_balls_as_bowler', 'fantasy_score_bowling'
]
df = pd.read_excel(file_path, usecols=columns_needed, nrows=5000)


X = df[[
    'balls_bowled', 'runs_conceded', 'wickets_taken',
    'bowled_done', 'lbw_done', 'maidens', 'dot_balls_as_bowler'
]]
y = df['fantasy_score_bowling']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


poly = PolynomialFeatures(degree=3)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)


model = LinearRegression()
model.fit(X_train_poly, y_train)


y_pred = model.predict(X_test_poly)


r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2:.4f}")


plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, color='green', alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel("Actual Fantasy Score (Bowling)")
plt.ylabel("Predicted Fantasy Score (Bowling)")
plt.title(f"Polynomial Regression (R² = {r2:.2f})")
plt.grid(True)
plt.tight_layout()
plt.show()
