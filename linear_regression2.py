import pandas as pd
import numpy as np

df=pd.read_csv("C:\\Users\\Administrator\\Downloads\\mw_pw_profiles.txt",low_memory=False)

# OLS or closed form solution
class FantasyScorePredictor:
    def __init__(self, df, feature_columns, target_column):
        self.df = df
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.betas = None
    
    def prepare_data(self):
        data = self.df[self.feature_columns + [self.target_column]].values
        data = np.insert(data, 0, 1, axis=1)  # Add bias term
        
        train_size = int(0.8 * len(data))
        
        self.x_train = data[:train_size, :-1]
        self.y_train = data[:train_size, -1]
        self.x_test = data[train_size:, :-1]
        self.y_test = data[train_size:, -1]

    def train(self):
        self.betas = np.linalg.inv(np.dot(self.x_train.T,self.x_train)).dot(self.x_train.T).dot(self.y_train)

    def predict(self):
        return np.dot(self.x_test,self.betas)

    def evaluate(self):
        y_pred = self.predict()
        residuals = self.y_test - y_pred
        RSS = np.dot(residuals.T , residuals)

        mean = np.mean(self.y_test)
        total_diff = self.y_test - mean
        TSS = np.dot(total_diff.T , total_diff)

        r_squared = 1 - (RSS / TSS)
        return float(r_squared)
    
print("R2 value using OLS:")
fantasy_batting_score=FantasyScorePredictor(df,['runs_scored','player_out','balls_faced','fours_scored','sixes_scored','dot_balls_as_batsman','order_seen'],'fantasy_score_batting')
fantasy_batting_score.prepare_data()
fantasy_batting_score.train()
print(fantasy_batting_score.evaluate())


fantasy_bowling_score=FantasyScorePredictor(df,['catches_taken','run_out_direct','run_out_throw','stumpings_done','balls_bowled','runs_conceded','wickets_taken','bowled_done','lbw_done','maidens','dot_balls_as_bowler'],'fantasy_score_bowling')
fantasy_bowling_score.prepare_data()
fantasy_bowling_score.train()
print(fantasy_bowling_score.evaluate())

fantasy_total_score=FantasyScorePredictor(df,['runs_scored','player_out','balls_faced','fours_scored','sixes_scored','dot_balls_as_batsman','order_seen','catches_taken','run_out_direct','run_out_throw','stumpings_done','balls_bowled','runs_conceded','wickets_taken','bowled_done','lbw_done','maidens','dot_balls_as_bowler'],'fantasy_score_total')
fantasy_total_score.prepare_data()
fantasy_total_score.train()
print(fantasy_total_score.evaluate())


# batch gradient descent solution
class fantasyscorepredict_gd:
    def __init__(self,feature_columns, target_column,learning_rate,epochs):
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.learning_rate=learning_rate
        self.epochs=epochs
        self.betas=np.zeros(len(feature_columns)+1)

    def prepare_data(self):
        df[self.feature_columns] = (df[self.feature_columns] - df[self.feature_columns].mean()) / df[self.feature_columns].std()
        data = df[self.feature_columns + [self.target_column]].values
        data = np.insert(data, 0, 1, axis=1)  # Add bias term
        
        train_size = int(0.8 * len(data))
            
        self.x_train = data[:train_size, :-1]
        self.y_train = data[:train_size, -1]
        self.x_test = data[train_size:, :-1]
        self.y_test = data[train_size:, -1]

    def train(self):
        for i in range(self.epochs):
            predictions = np.dot(self.x_train, self.betas)
            errors = self.y_train - predictions
            gradients = -2 * np.dot(self.x_train.T, errors) / len(self.y_train)
            self.betas -= self.learning_rate * gradients

    def evaluate(self):
        y_pred = np.dot(self.x_test,self.betas)
        residuals = self.y_test - y_pred
        RSS = np.dot(residuals.T , residuals)

        mean = np.mean(self.y_test)
        total_diff = self.y_test - mean
        TSS = np.dot(total_diff.T , total_diff)

        r_squared = 1 - (RSS / TSS)
        return float(r_squared)
    
print("R2 value using batch gradient descent:")
fantasy_batting_score=fantasyscorepredict_gd(['runs_scored','player_out','balls_faced','fours_scored','sixes_scored','dot_balls_as_batsman','order_seen'],'fantasy_score_batting',0.01,200)
fantasy_batting_score.prepare_data()
fantasy_batting_score.train()
print(fantasy_batting_score.evaluate())

fantasy_bowling_score=fantasyscorepredict_gd(['catches_taken','run_out_direct','run_out_throw','stumpings_done','balls_bowled','runs_conceded','wickets_taken','bowled_done','lbw_done','maidens','dot_balls_as_bowler'],'fantasy_score_bowling',0.01,200)
fantasy_bowling_score.prepare_data()
fantasy_bowling_score.train()
print(fantasy_bowling_score.evaluate())

fantasy_total_score=fantasyscorepredict_gd(['runs_scored','player_out','balls_faced','fours_scored','sixes_scored','dot_balls_as_batsman','order_seen','catches_taken','run_out_direct','run_out_throw','stumpings_done','balls_bowled','runs_conceded','wickets_taken','bowled_done','lbw_done','maidens','dot_balls_as_bowler'],'fantasy_score_total',0.01,200)
fantasy_total_score.prepare_data()
fantasy_total_score.train()
print(fantasy_total_score.evaluate())