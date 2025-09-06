import pandas as pd
import json
from pathlib import Path

# Paths
BASE_DIR = Path("C:/Users/Jcravid/Downloads/Betting dashboard/backend").resolve()
MODEL_PATH = BASE_DIR / "src" / "models" / "poisson_epl.json"
FIXTURES_CSV = BASE_DIR / "src" / "fixtures.csv"
FIXTURES_FIXED = BASE_DIR / "src" / "fixtures_fixed.csv"

# Load model team names
with open(MODEL_PATH, "r", encoding="utf-8") as f:
    model = json.load(f)
model_teams = set(model["teams"].keys())

# Manual mapping of common mismatches
name_map = {
    "Manchester United": "Man United",
    "Manchester City": "Man City",
    "Nottingham Forest": "Nott'm Forest",
    "Tottenham Hotspur": "Tottenham",
    "Sheffield Utd": "Sheffield United",
    "West Ham United": "West Ham",
    "Wolverhampton": "Wolves",
    "Brighton & Hove Albion": "Brighton",
    "Aston Villa FC": "Aston Villa",
    "Luton Town": "Luton",
    "Crystal Palace FC": "Crystal Palace",
    "Chelsea FC": "Chelsea",
    "Liverpool FC": "Liverpool",
    "Arsenal FC": "Arsenal",
    "Fulham FC": "Fulham",
    "Everton FC": "Everton",
    "Newcastle United": "Newcastle",
    "Brentford FC": "Brentford",
    "Burnley FC": "Burnley",
    # Add more if needed
}

# Load fixtures
df = pd.read_csv(FIXTURES_CSV)

# Apply mappings
df["home"] = df["home"].apply(lambda x: name_map.get(x, x))
df["away"] = df["away"].apply(lambda x: name_map.get(x, x))

# Verify teams exist in model
invalid = set(df["home"]).union(set(df["away"])) - model_teams
if invalid:
    print("⚠️ Warning: Some teams still don’t match model keys:", invalid)
else:
    print("✅ All team names now match the model!")

# Save fixed fixtures
df.to_csv(FIXTURES_FIXED, index=False)
print(f"Fixtures cleaned and saved to: {FIXTURES_FIXED}")
