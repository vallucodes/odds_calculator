import os
from typing import Iterable, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import duckdb
import numpy as np
import pandas as pd


PARQUET_DIR = "/media/vallu/Storage/Coding/Own_projects/betting_model/vallu_scraper/data/parquet"
FEATURES_DIR = "/media/vallu/Storage/Coding/Own_projects/betting_model/vallu_scraper/data/features"
TEAM_ELO_DIR = os.path.join(FEATURES_DIR, "team_elo")

os.makedirs(TEAM_ELO_DIR, exist_ok=True)

START_ELO = 1500.0


def expected_score(elo_1: float, elo_2: float) -> float:
    return 1.0 / (1.0 + 10 ** ((elo_2 - elo_1) / 400.0))


def update_elo(
    elo_1: float, elo_2: float, res: int, k1: float, k2: float
) -> tuple[float, float]:
    ea = expected_score(elo_1, elo_2)
    elo_1_new = elo_1 + k1 * ((1 - res) - ea)
    elo_2_new = elo_2 + k2 * (res - (1 - ea))
    return elo_1_new, elo_2_new


def load_matches() -> pd.DataFrame:
    """Load base matches data in chronological order."""
    con = duckdb.connect()
    df = con.execute(
        f"""
        SELECT
            hltv_match_id,
            team1_name,
            team2_name,
            team1_score,
            team2_score
        FROM '{PARQUET_DIR}/matches.parquet'
        ORDER BY hltv_match_id ASC
    """
    ).df()
    con.close()

    df["result"] = np.where(df["team1_score"] > df["team2_score"], 0, 1)
    return df


def compute_team_elo_features(
    matches_df: pd.DataFrame, high_k: float, low_k: float, threshold: int
) -> pd.DataFrame:
    """
    Compute team1 / team2 Elo trajectories for a given (HIGH_K, LOW_K, THRESHOLD)
    configuration and return a dataframe with one row per match.
    """
    n = len(matches_df)
    team1_elo_vals = np.empty(n, dtype=float)
    team2_elo_vals = np.empty(n, dtype=float)

    unique_teams = pd.unique(
        matches_df[["team1_name", "team2_name"]].values.ravel()
    )
    elo_state = {
        team: {"elo": float(START_ELO), "games_played": 0} for team in unique_teams
    }

    for i, row in enumerate(matches_df.itertuples(index=False)):
        team1 = row.team1_name
        team2 = row.team2_name

        # Elo before this match (based only on prior matches)
        team1_elo_vals[i] = elo_state[team1]["elo"]
        team2_elo_vals[i] = elo_state[team2]["elo"]

        res = 0 if row.team1_score > row.team2_score else 1

        k1 = high_k if elo_state[team1]["games_played"] < threshold else low_k
        k2 = high_k if elo_state[team2]["games_played"] < threshold else low_k

        new_elo1, new_elo2 = update_elo(
            elo_state[team1]["elo"],
            elo_state[team2]["elo"],
            res,
            k1,
            k2,
        )
        elo_state[team1]["elo"] = new_elo1
        elo_state[team2]["elo"] = new_elo2
        elo_state[team1]["games_played"] += 1
        elo_state[team2]["games_played"] += 1

    out_df = matches_df[["hltv_match_id"]].copy()
    out_df["team1_elo"] = team1_elo_vals
    out_df["team2_elo"] = team2_elo_vals
    return out_df


def build_team_elo_features(high_k: int, low_k: int, threshold: int) -> None:
    """
    Build team-level Elo features with given K-factors and threshold,
    and save to features_team_elo_HIGH_K_LOW_K_THRESHOLD.parquet.
    """
    filename = f"features_team_elo_{high_k}_{low_k}_{threshold}.parquet"
    output_path = os.path.join(TEAM_ELO_DIR, filename)

    # Safety: if file already exists, skip (supports reruns / partial runs)
    if os.path.exists(output_path):
        print(f"⏭️ Skipping existing file {output_path}")
        return

    matches_df = load_matches()
    out_df = compute_team_elo_features(matches_df, high_k, low_k, threshold)
    out_df.to_parquet(output_path, index=False)
    print(f"✅ Saved {len(out_df)} rows → {output_path}")


def main() -> None:
    # Define parameter ranges similarly to features_map_elo.py
    high_k_values = list(range(50, 201, 20))  # 50–200 step 20
    low_k_values = list(range(10, 101, 10))   # 10–100 step 10
    threshold_values = list(range(5, 31, 2))  # 5–30 step 2

    combos: list[Tuple[int, int, int]] = []
    for h in high_k_values:
        for l in low_k_values:
            for t in threshold_values:
                filename = f"features_team_elo_{h}_{l}_{t}.parquet"
                path = os.path.join(TEAM_ELO_DIR, filename)
                if os.path.exists(path):
                    print(f"Already have {filename}, skipping in queue build.")
                    continue
                combos.append((h, l, t))

    total = len(combos)
    print(f"Total combinations to run: {total}")

    # Use multiple processes to parallelize over parameter combinations,
    # but always leave at least one core free.
    cpu_count = os.cpu_count() or 4
    max_workers = max(1, cpu_count - 1)
    print(f"Using up to {max_workers} processes.")

    if total == 0:
        return

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_params = {
            executor.submit(build_team_elo_features, h, l, t): (h, l, t)
            for (h, l, t) in combos
        }

        for i, future in enumerate(as_completed(future_to_params), start=1):
            h, l, t = future_to_params[future]
            try:
                future.result()
                print(f"[{i}/{total}] Done HIGH_K={h}, LOW_K={l}, THRESHOLD={t}")
            except Exception as e:
                print(f"[{i}/{total}] FAILED HIGH_K={h}, LOW_K={l}, THRESHOLD={t}: {e}")


if __name__ == "__main__":
    main()

