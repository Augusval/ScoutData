# tests/test_api.py
"""
Tests de la API REST (api_fastapi.py) via TestClient, SIN levantar server.
Cubre: contrato de entrada (404/422), contrato de SALIDA (response_model),
singleton cacheado (Depends + lru_cache), convergencia motor==HTTP,
y la segunda cabeza de CLUBES (alias normalization, 404 honesto en
cross-league, 422 post-canonicalizacion, response_model de clubes).

Fixture club_engine LOCAL (no toca conftest.py ni test_clubs.py -> los
23 tests actuales quedan intactos). Total esperado: 30 passed.
"""
import pytest

from src.engine.predictor import predict_match
from src.engine.predictor_clubs import (
    load_club_models_pure,
    load_club_data_pure,
    predict_club_match,
)


@pytest.fixture(scope="session")
def club_engine():
    """Motor de clubes cargado directamente (para el test de convergencia
    motor==HTTP). No usa get_club_engine de la API: son dos rutas de carga
    que deben converger al mismo numero."""
    mh, ma = load_club_models_pure()
    return {"model_home": mh, "model_away": ma, "df": load_club_data_pure()}


# =========================================================================
# SELECCIONES (v1.1, intactos)
# =========================================================================
def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"
    assert "club" in res.json()["heads"]   # v1.2: dos cabezas


def test_teams_lists_known_teams(client):
    res = client.get("/teams")
    assert res.status_code == 200
    teams = res.json()
    assert isinstance(teams, list) and len(teams) > 100
    assert "France" in teams and "Argentina" in teams


def test_predict_extreme_via_http(client):
    res = client.post("/v1/predict", json={"home": "France", "away": "Arameans Suryoye"})
    assert res.status_code == 200
    body = res.json()
    assert body["blend"]["parametric_pct"] == 100
    assert body["probabilities_pct"]["home_win"] > 80.0
    assert body["most_likely_score"] == "3-1"
    assert body["ranking"]["home"]["fifa"].startswith("#")
    assert body["ranking"]["away"]["points"] == 700


def test_predict_unknown_team_returns_404(client):
    res = client.post("/v1/predict", json={"home": "France", "away": "Club Fantasma FC"})
    assert res.status_code == 404
    assert "fuera del modelo" in res.json()["detail"]


def test_predict_same_team_returns_422(client):
    res = client.post("/v1/predict", json={"home": "France", "away": "France"})
    assert res.status_code == 422


def test_predict_missing_field_returns_422(client):
    res = client.post("/v1/predict", json={"home": "France"})
    assert res.status_code == 422


def test_engine_is_cached_singleton():
    from api_fastapi import get_engine
    assert get_engine() is get_engine()


def test_response_contract_has_all_declared_fields(client):
    body = client.post("/v1/predict",
                       json={"home": "France", "away": "Arameans Suryoye"}).json()
    for key in ("match", "ranking", "domain", "xg", "blend",
                "probabilities_pct", "most_likely_score"):
        assert key in body, f"falta bloque declarado: {key}"
    assert isinstance(body["match"]["neutral"], bool)
    assert isinstance(body["ranking"]["home"]["points"], float)
    assert isinstance(body["domain"]["home"]["flag"], str)
    assert isinstance(body["xg"]["final"]["home"], float)
    assert isinstance(body["blend"]["parametric_pct"], int)
    assert isinstance(body["probabilities_pct"]["home_win"], float)
    assert isinstance(body["most_likely_score"], str)


def test_response_model_filters_internal_fields(client):
    body = client.post("/v1/predict",
                       json={"home": "France", "away": "Arameans Suryoye"}).json()
    assert "_internal_debug" not in body


def test_convergence_motor_equals_api(client, engine):
    home, away = "France", "Arameans Suryoye"
    r = predict_match(home, away, False,
                      engine["model_home"], engine["model_away"],
                      engine["df"], engine["fifa_ranking"])
    api = client.post("/v1/predict", json={"home": home, "away": away}).json()
    assert api["xg"]["final"]["home"] == pytest.approx(r["xg_h"], abs=0.05)
    assert api["xg"]["final"]["away"] == pytest.approx(r["xg_a"], abs=0.05)
    assert api["probabilities_pct"]["home_win"] == pytest.approx(r["probs"]["home_win"], abs=0.5)
    assert api["blend"]["parametric_pct"] == pytest.approx(r["w"] * 100, abs=1.0)


# =========================================================================
# CLUBES (v1.2, nuevos)
# =========================================================================
def test_club_extreme_via_http(client):
    """Man City vs Norwich (colista): extremo intra-liga -> blend alto, favorito."""
    res = client.post("/v1/predict/club", json={"home": "Man City", "away": "Norwich"})
    assert res.status_code == 200
    body = res.json()
    assert body["blend"]["parametric_pct"] >= 80
    assert body["probabilities_pct"]["home_win"] > 75.0
    assert body["elo"]["diff"] > 300
    assert body["source"]["league"] == "Premier League"


def test_club_cross_league_returns_404(client):
    """Man City (PL) vs Barcelona (LaLiga): no hay puente en football-data -> 404
    honesto, NO 200 con numeros inventados."""
    res = client.post("/v1/predict/club", json={"home": "Man City", "away": "Barcelona"})
    assert res.status_code == 404
    assert "no existe en el historico" in res.json()["detail"]


def test_club_same_team_after_canonicalization_returns_422(client):
    """'Manchester City' vs 'Man City' son el mismo club tras normalizar -> 422,
    no 404 (eso seria 'par inexistente', que es mentir sobre la causa)."""
    res = client.post("/v1/predict/club", json={"home": "Manchester City", "away": "Man City"})
    assert res.status_code == 422
    assert "mismo equipo tras normalizar" in res.json()["detail"]


def test_club_alias_normalization(client):
    """'Manchester City' (API-Football) canonicaliza a 'Man City' (football-data)
    y predice. El response expone BOTH: home_input (lo que mandaste) y home
    (canonical), para transparencia."""
    res = client.post("/v1/predict/club", json={"home": "Manchester City", "away": "Norwich"})
    assert res.status_code == 200
    body = res.json()
    assert body["match"]["home_input"] == "Manchester City"
    assert body["match"]["home"] == "Man City"
    assert body["blend"]["parametric_pct"] >= 80   # mismo extremo que el lookup directo


def test_club_response_contract(client):
    body = client.post("/v1/predict/club",
                       json={"home": "Man City", "away": "Norwich"}).json()
    for key in ("match", "elo", "xg", "blend", "probabilities_pct",
                "most_likely_score", "source"):
        assert key in body, f"falta bloque declarado: {key}"
    assert isinstance(body["match"]["neutral"], bool)
    assert isinstance(body["elo"]["diff"], float)
    assert isinstance(body["xg"]["final"]["home"], float)
    assert isinstance(body["blend"]["parametric_pct"], int)
    assert isinstance(body["probabilities_pct"]["home_win"], float)
    assert isinstance(body["most_likely_score"], str)
    assert isinstance(body["source"]["date"], str)
    assert isinstance(body["source"]["league"], str)


def test_club_response_model_filters_internal(client):
    body = client.post("/v1/predict/club",
                       json={"home": "Man City", "away": "Norwich"}).json()
    assert "_internal_debug" not in body


def test_club_convergence_motor_equals_api(client, club_engine):
    """Motor puro (fixture club_engine) vs endpoint HTTP (get_club_engine lru_cache).
    Deben coincidir al decimal -> un solo cerebro de clubes, dos bocas."""
    home, away = "Man City", "Norwich"
    r = predict_club_match(home, away, False,
                           club_engine["model_home"], club_engine["model_away"],
                           club_engine["df"])
    assert r is not None
    api = client.post("/v1/predict/club", json={"home": home, "away": away}).json()
    assert api["xg"]["final"]["home"] == pytest.approx(r["xg_h"], abs=0.05)
    assert api["xg"]["final"]["away"] == pytest.approx(r["xg_a"], abs=0.05)
    assert api["probabilities_pct"]["home_win"] == pytest.approx(r["probs"]["home_win"], abs=0.5)
    assert api["blend"]["parametric_pct"] == pytest.approx(r["w"] * 100, abs=1.0)
    assert api["elo"]["diff"] == pytest.approx(r["elo_diff"], abs=0.5)