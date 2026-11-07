# tests/test_engine.py
"""
Tests del motor desacoplado (src/engine/predictor.py).
Verifican PROPIEDADES robustas (no numeros exactos fragiles) en los dos
extremos que ya validaste en la UI: matchup extremo (gating parametrico)
y matchup parejo (dominio del ML). Tambien banderas ISO2 y cara-a-cara.
Las banderas se comparan con escapes \\U (ASCII) a proposito: los emoji de
bandera son pares de codepoints que se corrompen al copiar/pegar, y un test
no puede fallar por la corrupcion del portapapeles, solo por el codigo.
"""
import pandas as pd
import pytest

from src.engine.predictor import (
    predict_match, get_flag, get_head_to_head, get_confederation,
)

# Codepoints ASCII-safe de las banderas (regional indicators F+R, E+S, A+R).
FLAG_FR = "\U0001F1EB\U0001F1F7"   # Francia
FLAG_ES = "\U0001F1EA\U0001F1F8"   # Espana
FLAG_AR = "\U0001F1E6\U0001F1F7"   # Argentina
FLAG_WHITE = "\U0001F3F3\uFE0F"    # bandera blanca (diaspora / desconocido)


def test_extreme_matchup_engages_parametric_layer(engine):
    """France vs amateur: el gating debe activar ~100% la capa parametrica."""
    r = predict_match("France", "Arameans Suryoye", False,
                      engine["model_home"], engine["model_away"],
                      engine["df"], engine["fifa_ranking"])
    assert r["w"] == pytest.approx(1.0, abs=0.01), "blend parametrico deberia ser ~100%"
    assert r["xg_h"] > 3.0, "xG local esperado alto en un extremo"
    assert r["xg_a"] < 1.0, "xG visitante esperado bajo"
    assert r["probs"]["home_win"] > 80.0, "Francia deberia ser favorito abrumador"
    assert r["probs"]["home_win"] > r["probs"]["away_win"]


def test_even_matchup_lets_ml_dominate(engine):
    """Argentina vs Spain (#2 vs #1): partido parejo -> el ML manda, la formula casi no pesa."""
    r = predict_match("Argentina", "Spain", False,
                      engine["model_home"], engine["model_away"],
                      engine["df"], engine["fifa_ranking"])
    assert r["w"] < 0.2, "en un parejo el blend parametrico debe ser bajo"
    # Coherencia futbolistica: sin favorito claro, empate alto, local con leve sesgo.
    assert 30.0 < r["probs"]["home_win"] < 55.0
    assert r["probs"]["draw"] > 20.0
    assert abs(r["probs"]["home_win"] - r["probs"]["away_win"]) < 20.0


def test_flags_are_iso2_robust():
    """Banderas derivadas de ASCII: mapeo correcto, sin depender de literales emoji."""
    assert get_flag("France") == FLAG_FR
    assert get_flag("Spain") == FLAG_ES
    assert get_flag("Argentina") == FLAG_AR
    assert get_flag("Arameans Suryoye") == FLAG_WHITE          # sin ISO2, a proposito
    assert get_flag("Equipo Que No Existe ZZZ") == FLAG_WHITE    # desconocido seguro


def test_flag_is_valid_regional_indicator_pair():
    """Propiedad estructural: una bandera valida = 2 chars en rango regional indicator."""
    f = get_flag("France")
    assert len(f) == 2
    assert all(0x1F1E6 <= ord(c) <= 0x1F1FF for c in f)
    # Unicidad: dos paises distintos no pueden compartir bandera.
    assert get_flag("Spain") != get_flag("France")


def test_confederation_metadata():
    assert get_confederation("France") == "UEFA"
    assert get_confederation("Argentina") == "CONMEBOL"
    assert get_confederation("Arameans Suryoye") == "No FIFA"


def test_head_to_head_shape(engine):
    """Cara-a-cara devuelve un DataFrame con las columnas esperadas (solo pasado filtrado)."""
    h2h = get_head_to_head("Argentina", "Brazil", engine["df"], n=6)
    assert isinstance(h2h, pd.DataFrame)
    assert set(["Fecha", "Local", "Marcador", "Visitante", "Ganador"]).issubset(h2h.columns)
    if not h2h.empty:
        assert all("-" in m for m in h2h["Marcador"])