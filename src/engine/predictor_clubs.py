# src/engine/predictor_clubs.py
"""
Segunda cabeza del Hybrid Ensemble: modelo de CLUBES (intra-liga).
================================================================
Predice partidos de clubes usando features point-in-time generadas por
create_club_features.py y modelos entrenados por train_club_model.py.

Filosofia:
  - Reusa el espiritu del ensemble de selecciones.
  - No inventa numeros para matchups que la fuente no puede comparar.
  - Cross-league -> None (caption honesto en UI/API).
  - ELO intra-liga only (football-data no tiene European comps).

Uso:
  from src.engine.predictor_clubs import (
      load_club_models_pure,
      load_club_data_pure,
      predict_club_match,
  )
"""
import sys
import math
from pathlib import Path

import joblib
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

CLUB_MODEL_HOME_PATH = PROJECT_DIR / "src" / "models" / "xgboost_club_home.joblib"
CLUB_MODEL_AWAY_PATH = PROJECT_DIR / "src" / "models" / "xgboost_club_away.joblib"
CLUB_DATA_PATH = PROJECT_DIR / "data" / "processed" / "club_matches_with_features.csv"

CLUB_FEATURE_COLUMNS = [
    "home_elo", "away_elo", "elo_diff",
    "home_off_form", "away_off_form", "off_form_diff",
    "home_def_form", "away_def_form", "def_form_diff",
    "home_strength", "away_strength", "strength_diff",
    "home_matches_played", "away_matches_played", "experience_ratio",
    "home_tier", "away_tier",
    "neutral",
]

# Calibracion de clubes:
# selecciones usa FIFA points con divisor ~400.
# clubes usa ELO intra-liga con distribucion real std ~131 y extremo ~+/-460.
# Para que el extremo intra-liga genere una magnitud parametrica comparable,
# usamos divisor 150 (equivalencia: 1259/400 ~= 460/150).
ELO_DIVISOR = 150.0
BLEND_SATURATION = 3.0
BASE_HOME_GOALS = 1.45
BASE_AWAY_GOALS = 1.10
NEUTRAL_BASE = 1.28
ELO_SENSITIVITY = 0.25


def load_club_models_pure():
    if not CLUB_MODEL_HOME_PATH.exists():
        raise FileNotFoundError(
            f"Falta {CLUB_MODEL_HOME_PATH}. Corre primero: python src/data/train_club_model.py"
        )
    if not CLUB_MODEL_AWAY_PATH.exists():
        raise FileNotFoundError(
            f"Falta {CLUB_MODEL_AWAY_PATH}. Corre primero: python src/data/train_club_model.py"
        )
    return joblib.load(CLUB_MODEL_HOME_PATH), joblib.load(CLUB_MODEL_AWAY_PATH)


def load_club_data_pure() -> pd.DataFrame:
    if not CLUB_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Falta {CLUB_DATA_PATH}. Corre primero: python src/data/create_club_features.py"
        )
    df = pd.read_csv(CLUB_DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date", ascending=False).reset_index(drop=True)


def _load_poisson_fn():
    """
    Intenta reusar calculate_poisson_probabilities del motor de selecciones.
    Si no existe o tiene otro nombre, usa una implementacion local equivalente.
    """
    try:
        from src.engine.predictor import calculate_poisson_probabilities as fn
        if callable(fn):
            return fn
    except Exception:
        pass

    from scipy.stats import poisson

    def _local_poisson_probs(xg_h: float, xg_a: float) -> dict:
        max_goals = 10
        ph = poisson.pmf(list(range(max_goals + 1)), xg_h)
        pa = poisson.pmf(list(range(max_goals + 1)), xg_a)

        home = draw = away = 0.0
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                p = float(ph[i] * pa[j])
                if i > j:
                    home += p
                elif i == j:
                    draw += p
                else:
                    away += p

        total = home + draw + away
        if total <= 0:
            return {"home_win": 0.0, "draw": 0.0, "away_win": 0.0}

        return {
            "home_win": home / total * 100.0,
            "draw": draw / total * 100.0,
            "away_win": away / total * 100.0,
        }

    return _local_poisson_probs


POISSON_FN = _load_poisson_fn()


def parametric_xg_clubs(elo_home: float, elo_away: float, neutral: bool) -> tuple:
    """
    Capa parametrica para clubes, calibrada a ELO intra-liga.
    Misma logica conceptual que selecciones, pero con divisor 150.
    """
    bh, ba = (NEUTRAL_BASE, NEUTRAL_BASE) if neutral else (BASE_HOME_GOALS, BASE_AWAY_GOALS)
    adv = (elo_home - elo_away) / ELO_DIVISOR
    xg_h = max(0.05, bh * math.exp(ELO_SENSITIVITY * adv))
    xg_a = max(0.05, ba * math.exp(-ELO_SENSITIVITY * adv))
    return xg_h, xg_a


def blend_weight_clubs(elo_home: float, elo_away: float) -> float:
    """
    Gating por magnitud del matchup, calibrado a ELO de clubes.
    diff=0 -> 0% parametrico (manda ML)
    diff extremo intra-liga -> cerca de 100% parametrico
    """
    return min(1.0, max(0.0, abs(elo_home - elo_away) / ELO_DIVISOR / BLEND_SATURATION))


def predict_club_match(home_team: str, away_team: str, neutral: bool,
                       model_home, model_away, df: pd.DataFrame) -> dict | None:
    """
    Predice un matchup de clubes si existe en el dataset historico.

    Devuelve None si:
      - home == away
      - el par no existe en esa direccion
      - la fuente no puede comparar esos clubes (cross-league, etc.)

    Esto es clave: no inventamos un numero para lo que el modelo no conoce.
    La UI/API mostrara caption honesto.
    """
    if not home_team or not away_team or home_team == away_team:
        return None

    mask = (df["HomeTeam"] == home_team) & (df["AwayTeam"] == away_team)
    rows = df[mask]

    if rows.empty:
        return None

    row = rows.iloc[0]  # df esta ordenado por Date descendente -> ultimo encuentro

    feature_dict = {c: row[c] for c in CLUB_FEATURE_COLUMNS}
    inp = pd.DataFrame([feature_dict])[CLUB_FEATURE_COLUMNS]

    ml_h = max(0.0, float(model_home.predict(inp)[0]))
    ml_a = max(0.0, float(model_away.predict(inp)[0]))

    elo_h = float(row["home_elo"])
    elo_a = float(row["away_elo"])

    par_h, par_a = parametric_xg_clubs(elo_h, elo_a, neutral)
    w = blend_weight_clubs(elo_h, elo_a)

    xg_h = (1.0 - w) * ml_h + w * par_h
    xg_a = (1.0 - w) * ml_a + w * par_a

    probs = POISSON_FN(xg_h, xg_a)

    return {
        "home": home_team,
        "away": away_team,
        "neutral": bool(neutral),
        "ml_h": ml_h,
        "ml_a": ml_a,
        "par_h": par_h,
        "par_a": par_a,
        "w": w,
        "xg_h": xg_h,
        "xg_a": xg_a,
        "probs": probs,
        "elo_h": elo_h,
        "elo_a": elo_a,
        "elo_diff": elo_h - elo_a,
        "date": str(row["Date"].date()),
        "league": row["league"],
    }