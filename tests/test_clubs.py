# tests/test_clubs.py
"""
Tests del motor de CLUBES (src/engine/predictor_clubs.py), puro, sin HTTP.
Verifican PROPIEDADES robustas (no numeros exactos fragiles ni pares hardcodeados
mas alla de los que el smoke ya confirmo que existen en el CSV historico).

La convergencia motor==HTTP de clubes NO va aca: el endpoint /v1/predict/club
es pieza 5b. Aca solo el cerebro de clubes.

Fixture club_engine LOCAL (no toca conftest.py -> los 16 tests de selecciones
quedan intactos). Total esperado tras este archivo: 23 passed.
"""
import pytest

from src.engine.predictor_clubs import (
    load_club_models_pure,
    load_club_data_pure,
    predict_club_match,
    blend_weight_clubs,
    parametric_xg_clubs,
)


@pytest.fixture(scope="session")
def club_engine():
    """Motor de clubes ya cargado (modelos + df historico), compartido por tests."""
    mh, ma = load_club_models_pure()
    return {"model_home": mh, "model_away": ma, "df": load_club_data_pure()}


# --- 1. EXTREMO intra-liga: el gating parametrico se activa (verificado en smoke) ---
def test_club_extreme_matchup_engages_parametric_layer(club_engine):
    """Man City vs Norwich (colista): blend alto, favorito abrumador, xG extremo."""
    r = predict_club_match("Man City", "Norwich", False,
                           club_engine["model_home"], club_engine["model_away"],
                           club_engine["df"])
    assert r is not None, "el par existe en el historico (smoke lo confirmo)"
    assert r["w"] >= 0.80, "extremo intra-liga debe activar >=80% parametrico"
    assert r["probs"]["home_win"] >= 75.0, "lider vs colista -> favorito abrumador"
    assert r["xg_h"] > 2.0 and r["xg_a"] < 1.0, "xG extremo coherente"
    assert r["probs"]["home_win"] > r["probs"]["away_win"]


# --- 2. PROPIEDAD monotonica del gating (funcion pura, sin depender de pares) ---
def test_club_blend_weight_monotonic_with_elo_diff():
    """A mayor |elo_diff|, mayor peso parametrico; diff=0 -> 0%; extremo -> capped 1.0.
    Esta es la prueba REAL de la calibracion /150, no un partido hardcodeado."""
    assert blend_weight_clubs(1500, 1500) == pytest.approx(0.0), "parejo tecnico -> 0% param"
    assert blend_weight_clubs(1800, 1400) > blend_weight_clubs(1600, 1550), "monotonicidad"
    assert blend_weight_clubs(1400, 1800) == blend_weight_clubs(1800, 1400), "simetria en |diff|"
    assert blend_weight_clubs(2000, 1000) == pytest.approx(1.0), "extremo -> capped 1.0"
    # calibracion /150: diff 450 -> 450/450 = 1.0 (saturacion exacta)
    assert blend_weight_clubs(1725, 1275) == pytest.approx(1.0, abs=0.01)


# --- 3. parametric_xg de clubes: extrapolacion correcta en extremos ---
def test_club_parametric_xg_extrapolates():
    """La capa parametrica debe dar xG alto/bajo en extremos (lo que el ML no puede)."""
    h, a = parametric_xg_clubs(1800, 1300, False)   # diff 500 -> adv 3.33
    assert h > 2.5 and a < 0.8, "extremo -> xG local alto, visitante bajo"
    h2, a2 = parametric_xg_clubs(1500, 1500, False)  # diff 0 -> bases puras
    assert h2 == pytest.approx(1.45) and a2 == pytest.approx(1.10), "parejo -> bases"
    hn, an = parametric_xg_clubs(1500, 1500, True)   # neutral
    assert hn == pytest.approx(1.28) and an == pytest.approx(1.28), "neutral -> simetrico"


# --- 4. CROSS-LLEAGUE -> None (la limitacion intra-liga funcionando) ---
def test_club_cross_league_returns_none(club_engine):
    """Man City (PL) vs Barcelona (LaLiga): no hay puente en football-data -> None."""
    r = predict_club_match("Man City", "Barcelona", False,
                           club_engine["model_home"], club_engine["model_away"],
                           club_engine["df"])
    assert r is None, "cross-league no es comparable -> caption honesto, no numero inventado"


# --- 5. PAR INEXISTENTE -> None ---
def test_club_unknown_pair_returns_none(club_engine):
    r = predict_club_match("Man City", "Club Fantasma FC", False,
                           club_engine["model_home"], club_engine["model_away"],
                           club_engine["df"])
    assert r is None


# --- 6. MISMO EQUIPO -> None (guarda de sanidad) ---
def test_club_same_team_returns_none(club_engine):
    r = predict_club_match("Arsenal", "Arsenal", False,
                           club_engine["model_home"], club_engine["model_away"],
                           club_engine["df"])
    assert r is None


# --- 7. PROPIEDADES de probs y xG (coherencia matematica, no valores exactos) ---
def test_club_probs_and_xg_are_valid(club_engine):
    r = predict_club_match("Arsenal", "Chelsea", False,
                           club_engine["model_home"], club_engine["model_away"],
                           club_engine["df"])
    assert r is not None
    p = r["probs"]
    assert p["home_win"] + p["draw"] + p["away_win"] == pytest.approx(100.0, abs=0.5)
    assert all(v >= 0 for v in p.values()), "probabilidades no negativas"
    assert r["xg_h"] > 0 and r["xg_a"] > 0, "xG estrictamente positivo"
    assert 0.0 <= r["w"] <= 1.0, "blend en [0,1]"