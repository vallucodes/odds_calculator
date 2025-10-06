import pandas as pd
import numpy as np

class BettingOddsGenerator:
	def __init__(self, data: pd.DataFrame):
		"""
		data: DataFrame containing historical match data
		Expected columns: ['team_home', 'team_away', 'home_score', 'away_score']
		"""
		self.data = data
		self.team_stats = self.calculate_team_stats()

	def calculate_team_stats(self):
		"""Calculate basic stats like win rate, average goals, etc."""
		stats = {}
		teams = set(self.data['team_home']).union(self.data['team_away'])
		for team in teams:
			home_games = self.data[self.data['team_home'] == team]
			print("Home games for team", team)
			print(home_games)
			away_games = self.data[self.data['team_away'] == team]
			print("\nAway games for team", team)
			print(away_games)

			wins = ((home_games['home_score'] > home_games['away_score']).sum() +
					(away_games['away_score'] > away_games['home_score']).sum())
			total_games = len(home_games) + len(away_games)
			avg_goals_scored = (home_games['home_score'].sum() + away_games['away_score'].sum()) / total_games
			avg_goals_conceded = (home_games['away_score'].sum() + away_games['home_score'].sum()) / total_games

			stats[team] = {
				'win_rate': wins / total_games if total_games else 0,
				'avg_goals_scored': avg_goals_scored,
				'avg_goals_conceded': avg_goals_conceded
			}
		return stats

	def generate_odds(self, team_home, team_away):
		"""Generate odds for a match based on team stats"""
		home_stats = self.team_stats.get(team_home)
		away_stats = self.team_stats.get(team_away)

		if not home_stats or not away_stats:
			return None  # One of the teams has no data

		# Basic probabilistic model
		home_advantage = 0.1  # Slight advantage for home team
		home_prob = home_stats['win_rate'] + home_advantage
		away_prob = away_stats['win_rate']
		draw_prob = max(0, 1 - home_prob - away_prob)

		# Convert probabilities to decimal odds
		odds = {
			'home_win': round(1 / home_prob, 2) if home_prob else None,
			'draw': round(1 / draw_prob, 2) if draw_prob else None,
			'away_win': round(1 / away_prob, 2) if away_prob else None
		}
		return odds


# Example usage
data = pd.DataFrame({
	'team_home': ['A', 'B', 'A', 'C'],
	'team_away': ['B', 'C', 'C', 'A'],
	'home_score': [2, 1, 0, 1],
	'away_score': [1, 2, 1, 1]
})

generator = BettingOddsGenerator(data)
print(generator.generate_odds('A', 'C'))
