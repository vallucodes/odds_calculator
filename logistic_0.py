import pandas as pd
import numpy as np
import duckdb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import os
import matplotlib.pyplot as plt

SAVE_RESULTS = False
RESULTS_FILE = "/media/vallu/Storage/Coding/Own_projects/betting_model/model/results_log.csv"
RUN_NOTE = "logistic"

PARQUET_DIR = "/media/vallu/Storage/Coding/Own_projects/betting_model/vallu_scraper/data/parquet"
TEAM_ELO_FILE = (
    "/media/vallu/Storage/Coding/Own_projects/betting_model/"
    "vallu_scraper/data/features/team_elo/features_team_elo_110_20_13.parquet"
)
con = duckdb.connect()

df = con.execute(
    f"""
    SELECT hltv_match_id, team1_name, team2_name, team1_score, team2_score
    FROM '{PARQUET_DIR}/matches.parquet'
    ORDER BY hltv_match_id ASC
"""
).df()

df["result"] = np.where(df["team1_score"] > df["team2_score"], 0, 1)

# Games played per team (no Elo calculation here)
df["team1_games"] = 0
df["team2_games"] = 0
games_played = {
    team: 0
    for team in pd.unique(df[["team1_name", "team2_name"]].values.ravel())
}

for idx, row in df.iterrows():
    team1 = row["team1_name"]
    team2 = row["team2_name"]
    df.at[idx, "team1_games"] = games_played[team1]
    df.at[idx, "team2_games"] = games_played[team2]
    games_played[team1] += 1
    games_played[team2] += 1

# Merge precomputed team Elo features
if os.path.exists(TEAM_ELO_FILE):
    team_elo_df = pd.read_parquet(TEAM_ELO_FILE)
    required_team_cols = {"hltv_match_id", "team1_elo", "team2_elo"}
    missing_team_cols = required_team_cols - set(team_elo_df.columns)
    if missing_team_cols:
        raise ValueError(
            f"Missing expected team ELO columns in parquet: {sorted(missing_team_cols)}"
        )
    df = df.merge(
        team_elo_df[["hltv_match_id", "team1_elo", "team2_elo"]],
        on="hltv_match_id",
        how="left",
    )
else:
    raise FileNotFoundError(f"TEAM_ELO_FILE not found: {TEAM_ELO_FILE}")

# Filter to consider only >X matches played
filtered_df = df[(df["team1_games"] > 10) & (df["team2_games"] > 10)]

# Split remaining data chronologically
train_df, test_df = train_test_split(
    filtered_df,
    test_size=0.2,
    shuffle=False
)

num_rows = len(train_df)
print(f"Number of training matches: {num_rows}")

num_rows = len(test_df)
print(f"Number of testing matches: {num_rows}")

train_df = train_df.reset_index(drop=True)
test_df  = test_df.reset_index(drop=True)

# print(train_df.sample(50))

# Features (use team Elo from parquet)
feature_cols = ["team1_elo", "team2_elo"]
scaler = StandardScaler()
train_inputs = scaler.fit_transform(train_df[feature_cols])
test_inputs = scaler.transform(test_df[feature_cols])

# Target
train_targets = train_df['result']
test_targets = test_df['result']

model = LogisticRegression(solver='liblinear')
model.fit(train_inputs, train_targets)


# --- Feature importance report (coefficient-based) ---
coef = model.coef_[0]
coef_df = pd.DataFrame(
    {
        "feature": feature_cols,
        "coef": coef,
    }
)
coef_df["abs_coef"] = coef_df["coef"].abs()
coef_df = coef_df.sort_values("abs_coef", ascending=False)

print("\n=== Coefficient report (sorted by |coef|) ===")
print(coef_df.to_string(index=False, float_format="{:.4f}".format))

# Simple bar plot for the top-N strongest features
TOP_N = min(20, len(coef_df))
plot_df = coef_df.head(TOP_N).sort_values("coef")

plt.figure(figsize=(8, 5))
plt.barh(plot_df["feature"], plot_df["coef"])
plt.axvline(0, color="black", linewidth=1)
plt.title(f"Top {TOP_N} logistic regression coefficients")
plt.xlabel("Coefficient (effect on log-odds of team2 win)")
plt.tight_layout()
plt.show()

def predict_and_plot(inputs, targets, name=''):
    probs = model.predict_proba(inputs)[:, 1]

    ll = log_loss(targets, probs)
    auc = roc_auc_score(targets, probs)
    brier = brier_score_loss(targets, probs)

    print(f"{name} Log Loss:    {ll:.3f}  (baseline: 0.693)")
    print(f"{name} ROC-AUC:     {auc:.3f}  (baseline: 0.500)")
    print(f"{name} Brier Score: {brier:.3f} (baseline: 0.250)")

predict_and_plot(train_inputs, train_targets, name='Train')
predict_and_plot(test_inputs, test_targets, name='Test')

if SAVE_RESULTS:
    train_probs = model.predict_proba(train_inputs)[:, 1]
    test_probs  = model.predict_proba(test_inputs)[:, 1]

    # Build this run as a dict of metric -> value
    run_data = {
        'timestamp':     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'model':          RUN_NOTE,
    }
    # Add features one by one
    for i, feat in enumerate(feature_cols, 1):
        run_data[f'feature_{i}'] = feat

    # Add metrics
    run_data.update({
        'train_matches':  len(train_df),
        'test_matches':   len(test_df),
        'train_log_loss': round(log_loss(train_targets, train_probs), 4),
        'train_roc_auc':  round(roc_auc_score(train_targets, train_probs), 4),
        'train_brier':    round(brier_score_loss(train_targets, train_probs), 4),
        'test_log_loss':  round(log_loss(test_targets, test_probs), 4),
        'test_roc_auc':   round(roc_auc_score(test_targets, test_probs), 4),
        'test_brier':     round(brier_score_loss(test_targets, test_probs), 4),
    })

    # Convert to a single column Series (metric as index, value as data)
    new_col = pd.Series(run_data, name=datetime.now().strftime("%Y%m%d_%H%M%S"))

    if os.path.exists(RESULTS_FILE):
        existing = pd.read_csv(RESULTS_FILE, index_col=0)
        updated = pd.concat([existing, new_col], axis=1)
    else:
        updated = new_col.to_frame()

    # Find all feature rows across all runs
    all_features = [idx for idx in updated.index if idx.startswith('feature_')]
    # Sort them numerically
    all_features = sorted(all_features, key=lambda x: int(x.split('_')[1]))

    # Define fixed order: metadata first, then features, then metrics
    fixed_order = [
        'timestamp',
        'model',
        *all_features,
        'train_matches',
        'test_matches',
        'train_log_loss',
        'train_roc_auc',
        'train_brier',
        'test_log_loss',
        'test_roc_auc',
        'test_brier',
    ]

    # Reindex to enforce order, keeping any rows not in fixed_order at the end
    existing_rows = [r for r in fixed_order if r in updated.index]
    updated = updated.reindex(existing_rows)

    updated.to_csv(RESULTS_FILE)
