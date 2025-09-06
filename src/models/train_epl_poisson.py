"""
train_epl_poisson.py

Train a simple Poisson attack/defense model from an EPL CSV and save as JSON.

Usage (from backend folder, with venv active):
python src/models/train_epl_poisson.py --matches data/epl_matches.csv --out src/models/poisson_epl.json
"""

import argparse
from pathlib import Path
import json
import math
import pandas as pd

def detect_and_load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    mapping = {}

    # Map home team column
    for candidate in ("home", "hometeam", "home_team"):
        if candidate in cols:
            mapping[cols[candidate]] = "home"
            break
    # Map away team column
    for candidate in ("away", "awayteam", "away_team"):
        if candidate in cols:
            mapping[cols[candidate]] = "away"
            break
    # Map home goals column
    for candidate in ("home_goals", "fthg", "homegoals", "hg"):
        if candidate in cols:
            mapping[cols[candidate]] = "home_goals"
            break
    # Map away goals column
    for candidate in ("away_goals", "ftag", "awaygoals", "ag"):
        if candidate in cols:
            mapping[cols[candidate]] = "away_goals"
            break

    df = df.rename(columns=mapping)
    required = ["home", "away", "home_goals", "away_goals"]
    if not all(c in df.columns for c in required):
        raise SystemExit(f"CSV missing required columns. Found: {list(df.columns)}")

    df = df[required]
    df = df.dropna()
    return df

def train_poisson(matches_df: pd.DataFrame):
    # Global averages
    home = matches_df[["home","home_goals","away_goals"]].rename(columns={"home":"team","home_goals":"gf","away_goals":"ga"})
    away = matches_df[["away","away_goals","home_goals"]].rename(columns={"away":"team","away_goals":"gf","home_goals":"ga"})
    long = pd.concat([home, away], ignore_index=True)

    global_avg = long["gf"].mean()
    mean_home = matches_df["home_goals"].mean()
    mean_away = matches_df["away_goals"].mean()

    teams = {}
    for team in long["team"].unique():
        gf = long.loc[long["team"]==team, "gf"].mean()
        ga = long.loc[long["team"]==team, "ga"].mean()
        atk = math.log((gf+1e-6)/(global_avg+1e-6))
        dfc = math.log((ga+1e-6)/(global_avg+1e-6))
        teams[team] = {"gf": float(gf), "ga": float(ga), "attack": atk, "defense": dfc}

    home_adv = math.log((mean_home+1e-6)/(mean_away+1e-6))

    model = {"global_avg": float(global_avg), "home_adv": float(home_adv), "teams": teams}
    return model

def save_model(model: dict, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(model, f, indent=2)
    print(f"Saved model to {out_path}")

def main(matches_csv: str, out_json: str):
    matches_path = Path(matches_csv)
    out_path = Path(out_json)
    df = detect_and_load(matches_path)
    model = train_poisson(df)
    save_model(model, out_path)
    print("Global avg goals:", model["global_avg"])
    print("Home advantage (log):", model["home_adv"])
    for i, (team, vals) in enumerate(model["teams"].items()):
        if i < 10:
            print(team, "atk:", round(vals["attack"],3), "def:", round(vals["defense"],3))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--matches", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    main(args.matches, args.out)
