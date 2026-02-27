# Elo based prediction model. Extremely simple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# What is range of confidence used to evaluate model. 0.7 means only >0.7 confidence is considered
CONFIDENCE_MASK = 0.5

raw_df = pd.read_json('data/matches.json')

df = (
    raw_df[['hltv_match_id', 'team1_name', 'team2_name', 'team1_score', 'team2_score']]
      .sort_values(by='hltv_match_id', ascending=True)
      .drop(columns='hltv_match_id')
      .reset_index(drop=True)
)

train_df, test_df = train_test_split(df, test_size=0.2, shuffle=False)

unique_teams = pd.unique(df[['team1_name', 'team2_name']].values.ravel())

elo = {team: 1500 for team in unique_teams}

def print_elo_change(inspectTeam, team1, team2, elo_before, elo_after):
    if team1 == inspectTeam:
        opponent = team2
        old_elo = elo_before[team1]
        new_elo = elo_after[team1]
        old_elo_opp = elo_before[team2]
        new_elo_opp = elo_after[team2]
    elif team2 == inspectTeam:
        opponent = team1
        old_elo = elo_before[team2]
        new_elo = elo_after[team2]
        old_elo_opp = elo_before[team1]
        new_elo_opp = elo_after[team1]
    else:
        return

    print(f"{inspectTeam} vs {opponent}")
    print(f"Before match: {inspectTeam} {old_elo:.1f}, {opponent} {old_elo_opp:.1f}")
    print(f"After match:  {inspectTeam} {new_elo:.1f}, {opponent} {new_elo_opp:.1f}")
    print(f"Elo change:   {inspectTeam} {new_elo - old_elo:+.1f}, {opponent} {new_elo_opp - old_elo_opp:+.1f}")
    print("-" * 40)

def expected_score(elo_1, elo_2):
    return 1 / (1 + 10 ** ((elo_2 - elo_1) / 400))

def update_elo(elo_1, elo_2, res, k = 20):
    ea = expected_score(elo_1, elo_2)
    eb = 1 - ea
    elo_1_new = elo_1 + k * (res - ea)
    elo_2_new = elo_2 + k * ((1 - res) - eb)
    return elo_1_new, elo_2_new

inspectTeam = 'NAVI Junior'

for _, row in train_df.iterrows():
    team1, team2 = row['team1_name'], row['team2_name']
    score1, score2 = row['team1_score'], row['team2_score']

    if score1 > score2:
        res = 1
    elif score1 < score2:
        res = 0
    else:
        res = 0.5

    elo_before = elo.copy()

    elo[team1], elo[team2] = update_elo(elo[team1], elo[team2], res)

    # print_elo_change(inspectTeam, team1, team2, elo_before, elo)

# elo_df = pd.DataFrame(list(elo.items()), columns=['team', 'elo'])
# elo_ratings = elo_df.sort_values(by='elo', ascending=False).reset_index(drop=True)
# elo_ratings['elo'] = elo_ratings['elo'].round(0).astype(int)
# print(elo_ratings.head(50))

# Test: predict THEN update
predictions = []
actuals = []

for _, row in test_df.iterrows():
    team1, team2 = row['team1_name'], row['team2_name']

    prob_team1_wins = expected_score(elo[team1], elo[team2])
    predictions.append(prob_team1_wins)

    if score1 > score2:
        res = 1
    elif score1 < score2:
        res = 0
    else:
        res = 0.5

    actuals.append(res)
    elo[team1], elo[team2] = update_elo(elo[team1], elo[team2], res)

predictions = np.array(predictions)
actuals = np.array(actuals)

conf_mask = (predictions >= CONFIDENCE_MASK) | (predictions <= CONFIDENCE_MASK)
conf_mask &= (actuals != 0.5)

conf_accuracy = (
    (predictions[conf_mask] > 0.5) == (actuals[conf_mask] == 1)
).mean()

print(f"Confident accuracy: {conf_accuracy:.3f}")
