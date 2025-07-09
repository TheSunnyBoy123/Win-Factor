import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
import matplotlib.pyplot as plt

# Load and preprocess data
df = pd.read_csv('mw_pw_profiles.txt', low_memory=False)

base_features = ['runs_scored','balls_faced','fours_scored','sixes_scored','wickets_taken', 'runs_conceded', 'balls_bowled']

df['batting_strike_rate'] = (df['runs_scored'] / df['balls_faced'].replace(0, np.nan)) * 100
df['bowling_economy'] = (df['runs_conceded'] / (df['balls_bowled'].replace(0, np.nan) / 6))

df.fillna(0, inplace=True)

features = base_features + ['batting_strike_rate', 'bowling_economy']
target = 'fantasy_score_total'

df = df[features + [target]].copy()

x = df[features].values
y = df[target].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Decision Tree Class
class DecisionTree:
    def __init__(self, depth=3, min_samples_split=5):
        self.depth = depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, x, y, depth=0):
        n_samples, n_features = x.shape

        if depth >= self.depth or n_samples < self.min_samples_split:
            return np.mean(y)

        best_feature, best_thresh, best_mse = None, None, float('inf')
        for feature in range(n_features):
            thresholds = np.percentile(x[:, feature], np.linspace(0, 100, 20))
            thresholds = np.unique(thresholds)

            for thresh in thresholds:
                left_mask = x[:, feature] <= thresh
                right_mask = x[:, feature] > thresh

                if len(y[left_mask]) == 0 or len(y[right_mask]) == 0:
                    continue

                mse = (np.var(y[left_mask]) * len(y[left_mask])+ np.var(y[right_mask]) * len(y[right_mask])) / n_samples

                if mse < best_mse:
                    best_feature = feature
                    best_thresh = thresh
                    best_mse = mse
        print(f"Depth: {depth}, Samples: {n_samples}")

        if best_thresh is None:
            return np.mean(y)

        left_mask = x[:, best_feature] <= best_thresh
        right_mask = x[:, best_feature] > best_thresh

        return {
            'feature': best_feature,
            'threshold': best_thresh,
            'left': self.fit(x[left_mask], y[left_mask], depth + 1),
            'right': self.fit(x[right_mask], y[right_mask], depth + 1),
        }

    def predict_sample(self, X, node):
        if not isinstance(node, dict):
            return node
        if X[node['feature']] <= node['threshold']:
            return self.predict_sample(X, node['left'])
        else:
            return self.predict_sample(X, node['right'])

    def predict(self, x):
        return np.array([self.predict_sample(X, self.tree) for X in x])

    def train(self, x, y):
        self.tree = self.fit(x, y)

# Train and test
model = DecisionTree(depth=6, min_samples_split=5)
model.train(x_train, y_train)

preds = model.predict(x_test[:5])
print("Predictions: ", preds)
print("Actual     :", y_test[:5])
