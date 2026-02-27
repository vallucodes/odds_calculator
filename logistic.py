import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, brier_score_loss


# Data split
CALIBRATION_SET = 0.5
TRAIN_SET = 0.4
EVAL_SET = 0.1

# K parameters: How much one match infuences ELO change
# HIGH_K: Used for ELO calibration
# LOW_K: Used after ELO is calibrated
# THRESHOLD: Matches after which to switch to LOW_K
HIGH_K = 100
LOW_K = 20
THRESHOLD = 30

raw_df = pd.read_json('data/matches.json')

df = (
    raw_df[['hltv_match_id', 'team1_name', 'team2_name', 'team1_score', 'team2_score']]
      .sort_values(by='hltv_match_id', ascending=True)
      .drop(columns='hltv_match_id')
      .reset_index(drop=True)
)
df['result'] = np.where(df['team1_score'] > df['team2_score'], 1, 0)
df['team1_elo'] = None
df['team2_elo'] = None
df['team1_games'] = None
df['team2_games'] = None

# Init ELO to 1500 in separate dictionary
unique_teams = pd.unique(df[['team1_name', 'team2_name']].values.ravel())
elo = {
    team: {
        'elo': 1500,
        'games_played': 0
    }
    for team in unique_teams
}

def expected_score(elo_1, elo_2):
    return 1 / (1 + 10 ** ((elo_2 - elo_1) / 400))

def update_elo(elo_1, elo_2, res, k1, k2):
    ea = expected_score(elo_1, elo_2)
    eb = 1 - ea
    elo_1_new = elo_1 + k1 * (res - ea)
    elo_2_new = elo_2 + k2 * ((1 - res) - eb)
    return elo_1_new, elo_2_new

for index, row in df.iterrows():
    team1, team2 = row['team1_name'], row['team2_name']
    score1, score2 = row['team1_score'], row['team2_score']

    # Store current ELO
    df.at[index, 'team1_elo'] = elo[team1]['elo']
    df.at[index, 'team2_elo'] = elo[team2]['elo']
    df.at[index, 'team1_games'] = elo[team1]['games_played']
    df.at[index, 'team2_games'] = elo[team2]['games_played']

    res = df.at[index, 'result']

    # Set k
    k1 = HIGH_K if elo[team1]['games_played'] < THRESHOLD else LOW_K
    k2 = HIGH_K if elo[team2]['games_played'] < THRESHOLD else LOW_K

    elo[team1]['elo'], elo[team2]['elo'] = update_elo(elo[team1]['elo'], elo[team2]['elo'], res, k1, k2)

    # Increment games played
    elo[team1]['games_played'] += 1
    elo[team2]['games_played'] += 1


import matplotlib.pyplot as plt

# # After the loop, select teams (replace with your desired teams, e.g., based on final ELO or manually)
# final_elo_sorted = sorted(
#     elo.items(),
#     key=lambda x: x[1]['elo'],
#     reverse=True
# )[500:520]

# selected_teams = [team for team, _ in final_elo_sorted]
# print(f"Selected teams: {selected_teams}")

# # Collect ELO history for each selected team
# plt.figure(figsize=(12, 6))
# for team in selected_teams:
#     # Find all matches involving the team
#     mask_team1 = df['team1_name'] == team
#     mask_team2 = df['team2_name'] == team

#     # Get pre-match ELO and match indices
#     elo_history = []
#     match_indices = []

#     # From team1 perspective
#     elo_history.extend(df.loc[mask_team1, 'team1_elo'].tolist())
#     match_indices.extend(df[mask_team1].index.tolist())

#     # From team2 perspective
#     elo_history.extend(df.loc[mask_team2, 'team2_elo'].tolist())
#     match_indices.extend(df[mask_team2].index.tolist())

#     # Sort by match index to maintain chronological order
#     sorted_data = sorted(zip(match_indices, elo_history))
#     match_indices, elo_history = zip(*sorted_data) if sorted_data else ([], [])

#     # Plot the line
#     if match_indices:
#         plt.plot(match_indices, elo_history, marker='o', label=team)

# # Customize plot
# plt.title('ELO Development Over Matches for Selected Teams')
# plt.xlabel('Match Index (Chronological)')
# plt.ylabel('Pre-Match ELO Rating')
# plt.legend(title='Teams', bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# Print full table of every team with ELO
# elo_df = (
#     pd.DataFrame(
#         [(team, data['elo'], data['games_played']) for team, data in elo.items()],
#         columns=['team', 'elo', 'games_played']
#     )
#     .sort_values('elo', ascending=False)
#     .reset_index(drop=True)
# )
# print(elo_df.to_string(index=False))

# Filter to consider only >X matches played
filtered_df = df[(df['team1_games'] > 30) & (df['team2_games'] > 30)]

# Split remaining data chronologically
train_df, test_df = train_test_split(
    filtered_df,
    test_size=0.2,
    shuffle=False
)

num_rows = len(train_df)
print(f"Number of training matches: {num_rows}")

num_rows = len(test_df)
print(f"Number of training matches: {num_rows}")

train_df = train_df.reset_index(drop=True)
test_df  = test_df.reset_index(drop=True)

# print(train_df.sample(50))

# Features
feature_cols = ['team1_elo', 'team2_elo']
train_inputs = train_df[feature_cols]
test_inputs = test_df[feature_cols]

# Target
train_targets = train_df['result']
test_targets = test_df['result']

model = LogisticRegression(solver='liblinear')
model.fit(train_inputs, train_targets)


def predict_and_plot(inputs, targets, name=''):
    preds = model.predict(inputs)
    probs = model.predict_proba(inputs)[:, 1]

    acc = accuracy_score(targets, preds)
    ll = log_loss(targets, probs)
    auc = roc_auc_score(targets, probs)
    brier = brier_score_loss(targets, probs)

    print(f"{name} Accuracy:    {acc:.3f}")
    print(f"{name} Log Loss:    {ll:.3f}  (baseline: 0.693)")
    print(f"{name} ROC-AUC:     {auc:.3f}  (baseline: 0.500)")
    print(f"{name} Brier Score: {brier:.3f} (baseline: 0.250)")

predict_and_plot(train_inputs, train_targets, name='Train')
predict_and_plot(test_inputs, test_targets, name='Test')
