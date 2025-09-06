from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import pandas as pd
import json
from scipy.stats import poisson
import math
from typing import List

# -------------------------------
# Setup FastAPI app
# -------------------------------
app = FastAPI()

# Allow frontend calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Absolute directory paths
# -------------------------------
BASE_DIR = Path("C:/Users/Jcravid/Downloads/Betting dashboard/backend").resolve()
MODEL_PATH = BASE_DIR / "src" / "models" / "poisson_epl.json"
FIXTURES_CSV = BASE_DIR / "src" / "fixtures.csv"

# -------------------------------
# Request schema
# -------------------------------
class Fixture(BaseModel):
    home: str
    away: str

class FixturesRequest(BaseModel):
    fixtures: List[Fixture]

# -------------------------------
# Load model helper
# -------------------------------
def load_model():
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=500, detail="Model not trained. Run training first.")
    with open(MODEL_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

# -------------------------------
# Core prediction logic
# -------------------------------
def predict_fixture(home_team, away_team, model):
    global_avg = model["global_avg"]
    home_adv = model["home_adv"]

    if home_team not in model["teams"] or away_team not in model["teams"]:
        return None

    atk_home = model["teams"][home_team]["attack"]
    def_home = model["teams"][home_team]["defense"]
    atk_away = model["teams"][away_team]["attack"]
    def_away = model["teams"][away_team]["defense"]

    # Convert to λ
    lambda_home = max(0.01, (1 + atk_home) * (1 - def_away) * global_avg * (1 + home_adv))
    lambda_away = max(0.01, (1 + atk_away) * (1 - def_home) * global_avg)

    if math.isnan(lambda_home) or math.isnan(lambda_away):
        return None

    max_goals = 5
    home_probs = [poisson.pmf(i, lambda_home) for i in range(max_goals + 1)]
    away_probs = [poisson.pmf(i, lambda_away) for i in range(max_goals + 1)]

    home_win = sum(home_probs[i] * sum(away_probs[j] for j in range(i)) for i in range(max_goals + 1))
    draw = sum(home_probs[i] * away_probs[i] for i in range(max_goals + 1))
    away_win = sum(home_probs[i] * sum(away_probs[j] for j in range(i + 1, max_goals + 1)) for i in range(max_goals + 1))

    return {
        "home": home_team,
        "away": away_team,
        "lambda_home": round(lambda_home, 2),
        "lambda_away": round(lambda_away, 2),
        "home_win_prob": round(home_win, 2),
        "draw_prob": round(draw, 2),
        "away_win_prob": round(away_win, 2),
    }

# -------------------------------
# API endpoints
# -------------------------------

# ✅ Phase 3: POST /predict with custom fixtures
@app.post("/predict")
def get_predictions(request: FixturesRequest):
    model = load_model()
    predictions = []

    for fx in request.fixtures:
        result = predict_fixture(fx.home, fx.away, model)
        if result:
            predictions.append(result)

    if not predictions:
        raise HTTPException(status_code=500, detail="No valid predictions. Check team names.")

    return {"predictions": predictions}

# Keep old GET endpoint for CSV testing
@app.get("/predict")
def get_predictions_from_csv():
    model = load_model()
    if not FIXTURES_CSV.exists():
        raise HTTPException(status_code=500, detail="Fixtures file not found. Create fixtures.csv first.")

    try:
        df_fixtures = pd.read_csv(FIXTURES_CSV)
        fixtures = df_fixtures.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading fixtures CSV: {str(e)}")

    predictions = []
    for fx in fixtures:
        home_team, away_team = fx.get("home"), fx.get("away")
        if not home_team or not away_team:
            continue
        result = predict_fixture(home_team, away_team, model)
        if result:
            predictions.append(result)

    if not predictions:
        raise HTTPException(status_code=500, detail="No valid fixtures found. Check team names in fixtures.csv.")

    return {"predictions": predictions}
