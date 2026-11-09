# api_fastapi.py
"""
ScoutData REST API - segundo frente del motor desacoplado.
================================================================
v1.2: dos cabezas (selecciones + clubes), mismo contrato de calidad.
  - Selecciones: /v1/predict (Depends + lru_cache + response_model)
  - Clubes:      /v1/predict/club (mismo patron; None -> 404 honesto,
                 same-team post-canonicalizacion -> 422)
Endpoints: /health, /teams, /teams/clubs, /v1/predict, /v1/predict/club.
Docs: http://127.0.0.1:8000/docs

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
from src.engine.predictor_clubs import (
    load_club_models_pure, load_club_data_pure,
    predict_club_match, canonical_club_name,
)

app = FastAPI(
    title="ScoutData API",
    version="1.2",
    description="Prediccion de futbol (Hybrid Ensemble). Dos cabezas: "
                "selecciones (FIFA/ELO) + clubes (ELO intra-liga, football-data).",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ---------------------------------------------------------------------------
# DEPENDENCIAS: motores cargados UNA sola vez (lru_cache), inyectados via Depends.
# Sin `global`, sin mutacion: testeable (cache_clear()) y seguro.
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


@lru_cache(maxsize=1)
def get_club_engine() -> dict:
    """Segunda cabeza: modelos + df de clubes. Si faltan .joblib/CSV,
    load_club_models_pure levanta FileNotFoundError con mensaje de setup
    (lru_cache NO cachea excepciones, asi que tras entrenar la proxima call anda)."""
    mh, ma = load_club_models_pure()
    return {
        "model_home": mh,
        "model_away": ma,
        "df": load_club_data_pure(),
    }


# ---------------------------------------------------------------------------
# CONTRATO DE ENTRADA (selecciones) + SALIDA (selecciones)  [v1.1, intacto]
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
    model_config = ConfigDict(extra="ignore")
    match: MatchMeta
    ranking: RankingBlock
    domain: DomainBlock
    xg: XgBlock
    blend: BlendBlock
    probabilities_pct: ProbsBlock
    most_likely_score: str


# ---------------------------------------------------------------------------
# CONTRATO DE ENTRADA (clubes) + SALIDA (clubes)  [v1.2, nuevo]
# ---------------------------------------------------------------------------
class ClubRequest(BaseModel):
    home: str
    away: str
    neutral: bool = False

    @field_validator("home", "away")
    @classmethod
    def _no_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("nombre de equipo vacio")
        return v.strip()


class ClubMatchMeta(BaseModel):
    home: str
    away: str
    home_input: str      # lo que mando el cliente (ej "Manchester City")
    away_input: str
    neutral: bool


class ClubEloPair(BaseModel):
    home: float
    away: float
    diff: float


class ClubXgTeam(BaseModel):
    home: float
    away: float


class ClubXgBlock(BaseModel):
    ml: ClubXgTeam
    parametric: ClubXgTeam
    final: ClubXgTeam


class ClubBlendBlock(BaseModel):
    parametric_pct: int
    ml_pct: int


class ClubProbsBlock(BaseModel):
    home_win: float
    draw: float
    away_win: float


class ClubSourceBlock(BaseModel):
    date: str            # ultimo cruce historico disponible (proxy point-in-time)
    league: str


class ClubResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    match: ClubMatchMeta
    elo: ClubEloPair
    xg: ClubXgBlock
    blend: ClubBlendBlock
    probabilities_pct: ClubProbsBlock
    most_likely_score: str
    source: ClubSourceBlock


# ---------------------------------------------------------------------------
# ENDPOINTS
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "service": "ScoutData API", "version": "1.2",
            "heads": ["national", "club"]}


@app.get("/teams")
def teams(eng: dict = Depends(get_engine)):
    """Equipos de SELECCIONES que el modelo puede puntuar."""
    return sorted(set(eng["df"]["home_team"].unique()))


@app.get("/teams/clubs")
def teams_clubs(eng: dict = Depends(get_club_engine)):
    """Equipos de CLUBES (intra-liga europeo) que la segunda cabeza puede puntuar.
    Nombres canonical de football-data.co.uk (ej 'Man City', no 'Manchester City')."""
    df = eng["df"]
    return sorted(set(df["HomeTeam"].unique()) | set(df["AwayTeam"].unique()))


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
        "_internal_debug": "este campo jamas debe salir por la API",
    }


@app.post("/v1/predict/club", response_model=ClubResponse)
def predict_club(req: ClubRequest, eng: dict = Depends(get_club_engine)):
    # 422 post-canonicalizacion: "Manchester City" vs "Man City" son el mismo club.
    home_can = canonical_club_name(req.home)
    away_can = canonical_club_name(req.away)
    if home_can == away_can:
        raise HTTPException(status_code=422,
                            detail="Local y visitante son el mismo equipo tras normalizar nombres "
                                   f"('{req.home}' y '{req.away}' -> '{home_can}').")

    r = predict_club_match(req.home, req.away, req.neutral,
                           eng["model_home"], eng["model_away"], eng["df"])

    # None -> 404 honesto: cross-league, Liga AR, Champions, club nuevo.
    # NO inventamos un numero para lo que la fuente no puede comparar.
    if r is None:
        raise HTTPException(status_code=404,
                            detail="El par no existe en el historico intra-liga de football-data "
                                   "(cross-league, Liga AR, Champions/Europa League o club nuevo). "
                                   "El modelo no inventa numeros. Usa GET /teams/clubs para ver "
                                   "los clubes comparables.")

    return {
        "match": {
            "home": r["home"], "away": r["away"],
            "home_input": r["home_input"], "away_input": r["away_input"],
            "neutral": r["neutral"],
        },
        "elo": {
            "home": round(r["elo_h"], 1),
            "away": round(r["elo_a"], 1),
            "diff": round(r["elo_diff"], 1),
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
        "source": {"date": r["date"], "league": r["league"]},
        "_internal_debug": "este campo jamas debe salir por la API",
    }