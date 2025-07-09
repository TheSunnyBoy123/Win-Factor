import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
df=pd.read_csv('C:/Users/KRITIN KHOWALA/Desktop/Win Factor/mw_pw_profiles.csv')

X = df[['runs_scored', 'balls_faced', 'fours_scored', 'sixes_scored','dot_balls_as_batsman']]
y = df['fantasy_score_batting']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

degree = 2
poly = PolynomialFeatures(degree=degree)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

model = LinearRegression()
model.fit(X_train_poly, y_train)

y_pred = model.predict(X_test_poly)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error for batting score: {mse:.4f}")
print(f"R² Score for batting score: {r2:.4f}")


X = df[['catches_taken','run_out_direct', 'run_out_throw', 'stumpings_done','balls_bowled', 'runs_conceded','wickets_taken', 'bowled_done', 'lbw_done', 'maidens','dot_balls_as_bowler']]
y = df['fantasy_score_bowling']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

degree = 3
poly = PolynomialFeatures(degree=degree)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

model = LinearRegression()
model.fit(X_train_poly, y_train)

y_pred = model.predict(X_test_poly)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error for bowling score: {mse:.4f}")
print(f"R² Score for bowling score: {r2:.4f}")

X = df[['catches_taken','run_out_direct', 'run_out_throw', 'stumpings_done','balls_bowled', 'runs_conceded','wickets_taken', 'bowled_done', 'lbw_done', 'maidens','dot_balls_as_bowler','runs_scored', 'balls_faced', 'fours_scored', 'sixes_scored','dot_balls_as_batsman']]
y = df['fantasy_score_total']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

degree = 3
poly = PolynomialFeatures(degree=degree)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

model = LinearRegression()
model.fit(X_train_poly, y_train)

y_pred = model.predict(X_test_poly)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error for total score: {mse:.4f}")
print(f"R² Score for total score: {r2:.4f}")