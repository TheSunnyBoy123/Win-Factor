import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


class FantasyScorePredictor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.model = LinearRegression()
        self.poly = None

    def load_data(self, columns_needed):
        df = pd.read_csv(self.file_path, usecols=columns_needed, nrows=5000)
        return df

    def prepare_features(self, df, target_column):
        X = df.drop(columns=[target_column])
        y = df[target_column]
        return X, y

    def train_model(self, X, y, degree=4):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.poly = PolynomialFeatures(degree=degree)
        X_train_poly = self.poly.fit_transform(X_train)
        X_test_poly = self.poly.transform(X_test)

        self.model.fit(X_train_poly, y_train)
        y_pred = self.model.predict(X_test_poly)
        r2 = r2_score(y_test, y_pred)

        return X_test, y_test, y_pred, r2

    def plot_results(self, y_test, y_pred, title):
        plt.figure(figsize=(8, 5))
        plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
        plt.xlabel("Actual Fantasy Score")
        plt.ylabel("Predicted Fantasy Score")
        plt.title(f"{title} (R²: {r2_score(y_test, y_pred):.2f})")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def predict_batting_scores(self):
        columns = [
            'runs_scored', 'player_out', 'balls_faced',
            'fours_scored', 'sixes_scored', 'dot_balls_as_batsman',
            'order_seen', 'fantasy_score_batting'
        ]
        df = self.load_data(columns)
        X, y = self.prepare_features(df, 'fantasy_score_batting')
        X_test, y_test, y_pred, r2 = self.train_model(X, y, degree=4)
        self.plot_results(y_test, y_pred, "Polynomial Regression (Batting)")

    def predict_bowling_scores(self):
        columns = [
            'balls_bowled', 'runs_conceded', 'wickets_taken',
            'bowled_done', 'lbw_done', 'maidens', 'dot_balls_as_bowler',
            'fantasy_score_bowling'
        ]
        df = self.load_data(columns)
        X, y = self.prepare_features(df, 'fantasy_score_bowling')
        X_test, y_test, y_pred, r2 = self.train_model(X, y, degree=3)
        self.plot_results(y_test, y_pred, "Polynomial Regression (Bowling)")


file_path = "C:\\Users\\Administrator\\Downloads\\mw_pw_profiles.txt"  
predictor = FantasyScorePredictor(file_path)
predictor.predict_batting_scores()
predictor.predict_bowling_scores()
