# tests/test_api.py
"""
Tests de la API REST (api_fastapi.py) via TestClient, SIN levantar server.
Cubre: contrato de entrada (404/422), contrato de SALIDA (response_model),
singleton cacheado (Depends + lru_cache), y la convergencia motor==HTTP.
"""
import pytest

from src.engine.predictor import predict_match


def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


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


# --- NUEVO: el motor es un singleton cacheado (Depends + lru_cache) ---------
def test_engine_is_cached_singleton():
    """get_engine() debe devolver la MISMA instancia en dos calls (no recarga .joblib)."""
    from api_fastapi import get_engine
    assert get_engine() is get_engine()


# --- NUEVO: response_model impone el contrato de SALIDA ---------------------
def test_response_contract_has_all_declared_fields(client):
    """La respuesta HTTP trae TODOS los bloques declarados, con los TIPOS correctos."""
    body = client.post("/v1/predict",
                       json={"home": "France", "away": "Arameans Suryoye"}).json()
    # presencia de bloques
    for key in ("match", "ranking", "domain", "xg", "blend",
                "probabilities_pct", "most_likely_score"):
        assert key in body, f"falta bloque declarado: {key}"
    # tipos anidados (el contrato no es solo presencia, es forma)
    assert isinstance(body["match"]["neutral"], bool)
    assert isinstance(body["ranking"]["home"]["points"], float)
    assert isinstance(body["ranking"]["diff_fifa_points"], float)
    assert isinstance(body["domain"]["home"]["flag"], str)
    assert isinstance(body["xg"]["final"]["home"], float)
    assert isinstance(body["blend"]["parametric_pct"], int)
    assert isinstance(body["probabilities_pct"]["home_win"], float)
    assert isinstance(body["most_likely_score"], str)


# --- NUEVO: response_model FILTRA lo que no esta declarado ------------------
def test_response_model_filters_internal_fields(client):
    """El handler devuelve _internal_debug; MatchResponse NO lo declara -> se filtra."""
    body = client.post("/v1/predict",
                       json={"home": "France", "away": "Arameans Suryoye"}).json()
    assert "_internal_debug" not in body, \
        "response_model deberia filtrar campos no declarados de la salida"


# --- LA PRUEBA DEL DESACOPLAMIENTO (convergencia) ---------------------------
def test_convergence_motor_equals_api(client, engine):
    """
    Motor puro (fixture `engine`) vs endpoint HTTP (que carga SU propio motor
    via get_engine). Deben coincidir al decimal -> un solo cerebro, dos bocas.
    response_model no altera los numeros, solo tipa/filtra, asi que converge.
    """
    home, away = "France", "Arameans Suryoye"
    r = predict_match(home, away, False,
                      engine["model_home"], engine["model_away"],
                      engine["df"], engine["fifa_ranking"])
    api = client.post("/v1/predict", json={"home": home, "away": away}).json()
    assert api["xg"]["final"]["home"] == pytest.approx(r["xg_h"], abs=0.05)
    assert api["xg"]["final"]["away"] == pytest.approx(r["xg_a"], abs=0.05)
    assert api["probabilities_pct"]["home_win"] == pytest.approx(r["probs"]["home_win"], abs=0.5)
    assert api["blend"]["parametric_pct"] == pytest.approx(r["w"] * 100, abs=1.0)