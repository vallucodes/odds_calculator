import pandas as pd
import numpy as np

raw_df = pd.read_json('data/matches.json')

df = (
    raw_df[['hltv_match_id', 'team1_name', 'team2_name', 'team1_score', 'team2_score']]
      .sort_values(by='hltv_match_id', ascending=True)
      .drop(columns='hltv_match_id')
      .reset_index(drop=True)
)

unique = pd.Series(df['team1_name'].tolist() + df['team2_name'].tolist()).unique()

# Sort team names so order doesn't matter
df_sorted = pd.DataFrame(
    np.sort(df[['team1_name', 'team2_name']], axis=1),
    columns=['team1_name', 'team2_name']
)

# Count matches for each unique pair
pair_counts = df_sorted.groupby(['team1_name', 'team2_name']).size().reset_index(name='unique_match_pair_count')

# Count how many pairs have the same number of matches
matches_summary = pair_counts['unique_match_pair_count'].value_counts().sort_index()

print(matches_summary)

