import pandas as pd
import numpy as np
from feast import (
    Entity,
    FeatureService,
    FeatureView,
    Field,
    FileSource,
    Project,
    PushSource,
    RequestSource,
)

from feast.on_demand_feature_view import on_demand_feature_view
from feast.types import String, Int64
from datetime import timedelta

from feast import ValueType

from feast.feature_logging import LoggingConfig
from feast.infra.offline_stores.file_source import FileLoggingDestination


#project
project = Project(name="Winfactor_ys", description="Sattaontop")

#file source
cricinfo = FileSource(
    path="/content/Win-Factor_yashjeet/submissions_Yashjeet/feast_repo/repo/feature_repo/data/cricinfo.parquet",
    event_timestamp_column="start_date"
)

#entity
player = Entity(
    name="player_id_ent",
    value_type=ValueType.STRING,
    description="player_id"
)

#featureview
player_features = FeatureView(
    name="player_features",
    entities=[player],
    ttl=timedelta(days=1),
    schema=[
        Field(name="player_id", dtype=String),
        Field(name="match_id", dtype=String),
        Field(name="gender", dtype=String),
        Field(name="balls_per_over", dtype=Int64),
        Field(name="match_type", dtype=String),
        Field(name="runs_scored", dtype=Int64),
        Field(name="balls_faced", dtype=Int64),
        Field(name="fours_scored", dtype=Int64),
        Field(name="sixes_scored", dtype=Int64),
        Field(name="catches_taken", dtype=Int64),
        Field(name="run_out_direct", dtype=Int64),
        Field(name="run_out_throw", dtype=Int64),
        Field(name="stumpings_done", dtype=Int64),
        Field(name="dot_balls_as_batsman", dtype=Int64),
        Field(name="order_seen", dtype=Int64),
        Field(name="balls_bowled", dtype=Int64),
        Field(name="runs_conceded", dtype=Int64),
        Field(name="wickets_taken", dtype=Int64),
        Field(name="bowled_done", dtype=Int64),
        Field(name="lbw_done", dtype=Int64),
        Field(name="maidens", dtype=Int64),
        Field(name="dot_balls_as_bowler", dtype=Int64),
        Field(name="player_team", dtype=String),
        Field(name="opposition_team", dtype=String),
        Field(name="unique_name", dtype=String),
        Field(name="fantasy_score_batting", dtype=Int64),
        Field(name="fantasy_score_bowling", dtype=Int64),
        Field(name="fantasy_score_total", dtype=Int64),
    ],
    source=cricinfo
)





###

@on_demand_feature_view(
    sources=[player_features],
    schema=[
        Field(name="boundaries_wma", dtype=Int64),
        Field(name="fielding_wma", dtype=Int64),
        Field(name="dots_wma", dtype=Int64),
        Field(name="dot_balls_as_batsman_percentage_wma", dtype=Int64),
        Field(name="batting_aggression_wma", dtype=Int64),
        Field(name="strike_rate_wma", dtype=Int64),
        Field(name="economy_wma", dtype=Int64),
        Field(name="runs_scored_wma", dtype=Int64),
        Field(name="runs_conceded_wma", dtype=Int64),
        Field(name="wickets_taken_wma", dtype=Int64),
        Field(name="player_role", dtype=Int64),
        Field(name="rolling_fantasy_batting", dtype=Int64),
        Field(name="rolling_fantasy_bowling", dtype=Int64),
        Field(name="rolling_fantasy_total", dtype=Int64),
    ],
)
def transf_new(cricinfo):
    df = pd.DataFrame(cricinfo)
    
    obj_cols = ["player_id", "match_id", "gender", "match_type", "player_team", "opposition_team", "unique_name"]
    for col in obj_cols:
        if col in df.columns:
            df[col] = df[col].dropna().astype(str)
    
    
    df["boundaries"] = df["sixes_scored"] + df["fours_scored"]
    df["fielding"] = df["run_out_direct"] + df["run_out_throw"] + df["stumpings_done"] + df["catches_taken"]
    df["dots"] = df["dot_balls_as_bowler"] + df["maidens"]*9
    df["dot_balls_as_batsman_percentage"] = (df["dot_balls_as_batsman"] / df["balls_faced"])
    df["batting_aggression"] = np.where(df.balls_faced != 0,
              (((2*df["sixes_scored"] + df["fours_scored"]) / df["balls_faced"])), 0)
    df["strike_rate"] = np.where(df.balls_faced != 0, ((df["runs_scored"] / df["balls_faced"]) * 100), 0)
    df["economy"] = np.where(df.balls_bowled != 0, ((df["runs_conceded"] / df["balls_bowled"]) * 100), 0)
    df = df.drop(['sixes_scored', 'fours_scored', 'run_out_direct', 'run_out_throw', 'stumpings_done', 'catches_taken', 'dot_balls_as_bowler', 'maidens'], axis=1)
    def identify_player_role(df_player):

        total_matches = len(df_player)
        # percentage of matches with non-zero bowls bowled
        bowling_matches = (df_player['balls_bowled'] > 0).sum()
        bowling_percentage = (bowling_matches / total_matches) if total_matches > 0 else 0

        # percentage of matches with non-zero balls faced
        batting_matches = (df_player['balls_faced'] > 0).sum()
        batting_percentage = (batting_matches / total_matches) if total_matches > 0 else 0

        #average order seen
        average_order_seen = df_player['order_seen'].mean()

        if bowling_percentage >= 0.70:
            if batting_percentage >= 0.60 and average_order_seen > 6.5: # Using 6.5 as the threshold for order seen
                return 3
            else:
                return 2
        else:
            return 1
    
    # 3 - allrounder
    # 2 - bowler
    # 1 - batsman
    ###
    
    


    player_roles = df.groupby('player_id').apply(identify_player_role).reset_index(name='player_role')

    df = df.merge(player_roles, on='player_id', how='right')
    
    df["player_role"] = df["player_role"].fillna(0).astype("int64")  # make 100% sure it's int
#    df["player_role"] = df["player_role"].fillna("Unknown").astype(str)
    df.drop('order_seen', axis=1, inplace=True)
    
    ###
    #WMA
    
    def calc_wma(df, span, w=(np.pi), cols=['batting_aggression','dot_balls_as_batsman_percentage','boundaries', 'fielding', 'dots', 'strike_rate', 'economy', 'runs_scored','runs_conceded','wickets_taken']):
        ## using the weights as exponents of the golden ratio
        df_wma = df.copy()
        weights = np.array([w**(span - i - 1) for i in range(span)])
        weights /= weights.sum() # Normalize weights

        for col in cols:
            if col in df_wma.columns:
                df_wma[f'{col}_wma'] = df_wma.groupby('player_id')[col].transform(
                    lambda x: x.rolling(window=span).apply(lambda y: np.dot(y, weights), raw=True)
                )
                df_wma.drop(col, axis=1, inplace=True)
        return df_wma
        
    ###
    
    df_wma = calc_wma(df,5)
    
    ###
    
    # ARITRA's FEATURE
    weights = [0.4,0.3,0.2,0.1]

    df_wma['rolling_fantasy_batting'] = df_wma.groupby('player_id')['fantasy_score_batting'].transform(
                lambda x: x.shift(1).rolling(window=4).apply(lambda y: np.dot(y, weights), raw=True)
            )
    df_wma['rolling_fantasy_bowling'] = df_wma.groupby('player_id')['fantasy_score_bowling'].transform(
                lambda x: x.shift(1).rolling(window=4).apply(lambda y: np.dot(y, weights), raw=True)
            )
    df_wma['rolling_fantasy_total'] = df_wma.groupby('player_id')['fantasy_score_total'].transform(
                lambda x: x.shift(1).rolling(window=4).apply(lambda y: np.dot(y, weights), raw=True)
            )
    
    ###

    columns_to_cast = [
        "boundaries_wma", "fielding_wma", "dots_wma",
        "dot_balls_as_batsman_percentage_wma", "batting_aggression_wma",
        "strike_rate_wma", "economy_wma", "runs_scored_wma",
        "runs_conceded_wma", "wickets_taken_wma",
        "rolling_fantasy_batting", "rolling_fantasy_bowling", "rolling_fantasy_total"
    ]
    

    
    for col in columns_to_cast:
        if col in df_wma.columns:
            df_wma[col] = df_wma[col].fillna(0).round().astype("int64")
    if "player_role" in df_wma.columns:
        df_wma["player_role"] = df_wma["player_role"].astype("int64")
        
    return df_wma[[
    "boundaries_wma", "fielding_wma", "dots_wma",
    "dot_balls_as_batsman_percentage_wma", "batting_aggression_wma",
    "strike_rate_wma", "economy_wma", "runs_scored_wma",
    "runs_conceded_wma", "wickets_taken_wma", "player_role",
    "rolling_fantasy_batting", "rolling_fantasy_bowling", "rolling_fantasy_total"
]]


#########
### | ###
#########


#Feature Service

fetch_features = FeatureService(
    name="fetch_features",
    features=[
        player_features, # Sub-selects a feature from a feature view
        transf_new  # Selects all features from the feature view
    ],
    logging_config=LoggingConfig(
        destination=FileLoggingDestination(path="/content/Win-Factor_yashjeet/submissions_Yashjeet/feast_repo/repo/feature_repo/data/logs")
    ),
)



