# tests/conftest.py
"""
Configuracion de pytest para ScoutData.
- Agrega la RAIZ del proyecto al sys.path para que `src.engine.*` y
  `api_fastapi` sean importables desde los tests (pytest, por defecto,
  solo pone tests/ en el path, no la raiz).
- Define fixtures de SESION: carga modelos + df + ranking UNA sola vez
  y los comparte entre todos los tests (rapido y sin recargar .joblib).
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.engine.predictor import (  # noqa: E402
    load_models_pure, load_data_pure, load_fifa_ranking_pure,
)


@pytest.fixture(scope="session")
def engine():
    """Motor ya cargado (modelos + dataset + ranking), compartido por todos los tests."""
    mh, ma = load_models_pure()
    return {
        "model_home": mh,
        "model_away": ma,
        "df": load_data_pure(),
        "fifa_ranking": load_fifa_ranking_pure(),
    }


@pytest.fixture(scope="session")
def client():
    """Cliente HTTP de la API, sin levantar un server real (TestClient de Starlette)."""
    from fastapi.testclient import TestClient
    from api_fastapi import app
    return TestClient(app)