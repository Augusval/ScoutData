# api_fastapi.py
"""
ScoutData REST API - segundo frente del motor desacoplado.
================================================================
Mejora #2: Depends (inyeccion de motor, sin global mutable) +
response_model (contrato de SALIDA tipado que filtra campos internos).
Endpoints: /health, /teams, /v1/predict. Docs: http://127.0.0.1:8000/docs

Correr desde la RAIZ:  uvicorn api_fastapi:app --reload
"""
import sys
from pathlib import Path
from functools import lru_cache

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, field_validator

from src.engine.predictor import (
    load_models_pure, load_data_pure, load_fifa_ranking_pure,
    predict_match, get_flag, get_confederation, fmt_rank,
)

app = FastAPI(
    title="ScoutData API",
    version="1.1",
    description="Prediccion de futbol (Hybrid Ensemble: XGBoost + calibracion Elo/FIFA + Poisson).",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ---------------------------------------------------------------------------
# DEPENDENCIA: motor cargado UNA sola vez (lru_cache), inyectado via Depends.
# Sin `global`, sin mutacion: testeable (get_engine.cache_clear()) y seguro.
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_engine() -> dict:
    mh, ma = load_models_pure()
    return {
        "model_home": mh,
        "model_away": ma,
        "df": load_data_pure(),
        "fifa_ranking": load_fifa_ranking_pure(),
    }


# ---------------------------------------------------------------------------
# CONTRATO DE ENTRADA (ya lo tenias) + CONTRATO DE SALIDA (nuevo)
# ---------------------------------------------------------------------------
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


class TeamRanking(BaseModel):
    fifa: str
    points: float


class RankingBlock(BaseModel):
    home: TeamRanking
    away: TeamRanking
    diff_fifa_points: float


class DomainTeam(BaseModel):
    flag: str
    confederation: str


class DomainBlock(BaseModel):
    home: DomainTeam
    away: DomainTeam


class XgTeam(BaseModel):
    home: float
    away: float


class XgBlock(BaseModel):
    ml: XgTeam
    parametric: XgTeam
    final: XgTeam


class BlendBlock(BaseModel):
    parametric_pct: int
    ml_pct: int


class ProbsBlock(BaseModel):
    home_win: float
    draw: float
    away_win: float


class MatchMeta(BaseModel):
    home: str
    away: str
    neutral: bool


class MatchResponse(BaseModel):
    # extra="ignore" explícito: cualquier campo que el handler devuelva y NO
    # este aqui se FILTRA de la respuesta (demostrado en test_api).
    model_config = ConfigDict(extra="ignore")

    match: MatchMeta
    ranking: RankingBlock
    domain: DomainBlock
    xg: XgBlock
    blend: BlendBlock
    probabilities_pct: ProbsBlock
    most_likely_score: str


# ---------------------------------------------------------------------------
# ENDPOINTS
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "service": "ScoutData API", "version": "1.1"}


@app.get("/teams")
def teams(eng: dict = Depends(get_engine)):
    """Equipos que el modelo puede puntuar (para que un frontend sepa que mandar)."""
    return sorted(set(eng["df"]["home_team"].unique()))


@app.post("/v1/predict", response_model=MatchResponse)
def predict(req: MatchRequest, eng: dict = Depends(get_engine)):
    known = set(eng["df"]["home_team"].unique())

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
        # Campo INTERNO que NO esta en MatchResponse -> response_model lo filtra.
        # Lo dejamos a proposito para que un test demuestre el filtrado de salida.
        "_internal_debug": "este campo jamas debe salir por la API",
    }