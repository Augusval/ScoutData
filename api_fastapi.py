# api_fastapi.py
"""
ScoutData REST API - segundo frente del motor desacoplado.
===========================================================
Demuestra que src/engine/predictor.py (Python puro, sin Streamlit) sirve
igual para una API de produccion que para el dashboard. Misma logica,
dos consumidores. Endpoints: /health, /teams, /v1/predict.

Correr desde la RAIZ:  uvicorn api_fastapi:app --reload
Docs automaticas:      http://127.0.0.1:8000/docs
"""
import sys
from pathlib import Path

# Asegurar que la raiz este en el path (para `from src.engine...`)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

from src.engine.predictor import (
    load_models_pure, load_data_pure, load_fifa_ranking_pure,
    predict_match, get_flag, get_confederation, fmt_rank,
)

app = FastAPI(
    title="ScoutData API",
    version="1.0",
    description="Prediccion de futbol (Hybrid Ensemble: XGBoost + calibracion Elo/FIFA + Poisson).",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Singleton perezoso: carga modelos+df+ranking UNA sola vez, no por request.
_ENGINE = None
def _get_engine():
    global _ENGINE
    if _ENGINE is None:
        mh, ma = load_models_pure()
        _ENGINE = {"model_home": mh, "model_away": ma,
                   "df": load_data_pure(), "fifa_ranking": load_fifa_ranking_pure()}
    return _ENGINE


class MatchRequest(BaseModel):
    home: str
    away: str
    neutral: bool = False

    @field_validator("home", "away")
    @classmethod
    def _no_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("nombre de equipo vacio")
        return v.strip()


@app.get("/health")
def health():
    return {"status": "ok", "service": "ScoutData API", "version": "1.0"}


@app.get("/teams")
def teams():
    """Equipos que el modelo puede puntuar (para que un frontend sepa que mandar)."""
    return sorted(set(_get_engine()["df"]["home_team"].unique()))


@app.post("/v1/predict")
def predict(req: MatchRequest):
    eng = _get_engine()
    known = set(eng["df"]["home_team"].unique())

    # Validacion de dominio: equipo desconocido -> 404 explicito, no numero inventado.
    missing = [t for t in (req.home, req.away) if t not in known]
    if missing:
        raise HTTPException(status_code=404,
                            detail=f"Equipo(s) fuera del modelo: {missing}. Usa GET /teams.")
    if req.home == req.away:
        raise HTTPException(status_code=422, detail="Local y visitante no pueden ser el mismo equipo.")

    r = predict_match(req.home, req.away, req.neutral, **eng)

    return {
        "match": {"home": req.home, "away": req.away, "neutral": req.neutral},
        "ranking": {
            "home": {"fifa": fmt_rank(r["hf"]), "points": r["hf"]["points"]},
            "away": {"fifa": fmt_rank(r["af"]), "points": r["af"]["points"]},
            "diff_fifa_points": round(r["fifa_diff"], 1),
        },
        "domain": {
            "home": {"flag": get_flag(req.home), "confederation": get_confederation(req.home)},
            "away": {"flag": get_flag(req.away), "confederation": get_confederation(req.away)},
        },
        "xg": {
            "ml": {"home": round(r["ml_h"], 2), "away": round(r["ml_a"], 2)},
            "parametric": {"home": round(r["par_h"], 2), "away": round(r["par_a"], 2)},
            "final": {"home": round(r["xg_h"], 2), "away": round(r["xg_a"], 2)},
        },
        "blend": {"parametric_pct": round(r["w"] * 100), "ml_pct": round((1 - r["w"]) * 100)},
        "probabilities_pct": {
            "home_win": round(r["probs"]["home_win"], 1),
            "draw": round(r["probs"]["draw"], 1),
            "away_win": round(r["probs"]["away_win"], 1),
        },
        "most_likely_score": f"{round(r['xg_h'])}-{round(r['xg_a'])}",
    }
