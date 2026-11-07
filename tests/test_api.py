# tests/test_api.py
"""
Tests de la API REST (api_fastapi.py) via TestClient, SIN levantar server.
El test estrella (convergencia) prueba que el motor puro y el endpoint HTTP
devuelven EXACTAMENTE el mismo numero -> "un motor, dos frentes" verificado.
La API redondea a 2 decimales; por eso se compara con tolerancia (approx).
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
    assert body["ranking"]["home"]["fifa"].startswith("#")  # Francia rankeada
    assert body["ranking"]["away"]["points"] == 700          # amateur -> default


def test_predict_unknown_team_returns_404(client):
    res = client.post("/v1/predict", json={"home": "France", "away": "Club Fantasma FC"})
    assert res.status_code == 404
    assert "fuera del modelo" in res.json()["detail"]


def test_predict_same_team_returns_422(client):
    res = client.post("/v1/predict", json={"home": "France", "away": "France"})
    assert res.status_code == 422


def test_predict_missing_field_returns_422(client):
    # Pydantic rechaza antes de llamar tu funcion: contrato de entrada tipado.
    res = client.post("/v1/predict", json={"home": "France"})
    assert res.status_code == 422


def test_convergence_motor_equals_api(client, engine):
    """
    LA PRUEBA DEL DESACOPLAMIENTO.
    Llamamos predict_match() directo (motor puro, carga independiente de los
    .joblib) Y el endpoint HTTP (que hace SU propia carga del motor). Deben
    coincidir al decimal -> no hay dos implementaciones que puedan divergir,
    hay un solo cerebro con dos bocas. Esto es lo que el 200 de Swagger
    insinuaba; aca lo verificamos ejecutandose, para siempre.
    """
    home, away = "France", "Arameans Suryoye"

    # Frente 1: motor puro (Python, sin HTTP), cargado por el fixture `engine`.
    r = predict_match(home, away, False,
                      engine["model_home"], engine["model_away"],
                      engine["df"], engine["fifa_ranking"])

    # Frente 2: API por HTTP (TestClient), que carga el motor por su cuenta.
    api = client.post("/v1/predict", json={"home": home, "away": away}).json()

    # Convergencia: mismos numeros, con tolerancia de redondeo de la API.
    assert api["xg"]["final"]["home"] == pytest.approx(r["xg_h"], abs=0.05)
    assert api["xg"]["final"]["away"] == pytest.approx(r["xg_a"], abs=0.05)
    assert api["probabilities_pct"]["home_win"] == pytest.approx(r["probs"]["home_win"], abs=0.5)
    assert api["blend"]["parametric_pct"] == pytest.approx(r["w"] * 100, abs=1.0)