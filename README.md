# odds_calculator
Betting odds model

TODO
update features_map_winrate_lXX.py using features_map_winrate_l5.py as an example. Reasons: faster, order strictly by hltv_id.
use tester parquete_map_winrate.py to validate

update features_rating_kast_swing_lXX.py. Reasons: faster, order strictly by hltv_id.



now make new model in new file based this model. Make it sweep through all team elo calculation combinations, which are located as parquets in:

/media/vallu/Storage/Coding/Own_projects/betting_model/vallu_scraper/data/features/team_elo/

Example of one run is in logistic_3_l5.py

Make similar analysis output of best combination like is done in
logistic_4_l15_map_elo_sweep.py