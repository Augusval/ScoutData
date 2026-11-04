# src/models/train_model.py
"""
ScoutData AI - Modelo de Predicción de Goles Esperados (xG) con Ranking FIFA
=============================================================================
Entrena DOS modelos de regresión XGBoost (goles local y visitante)
y calcula probabilidades con Distribución de Poisson.
"""
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import joblib
from scipy.stats import poisson

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

PROJECT_DIR = Path(__file__).resolve().parents[2]
FEATURES_DATA_PATH = PROJECT_DIR / "data" / "processed" / "matches_with_features.csv"
MODEL_HOME_PATH = PROJECT_DIR / "src" / "models" / "xgboost_home_goals.joblib"
MODEL_AWAY_PATH = PROJECT_DIR / "src" / "models" / "xgboost_away_goals.joblib"

FEATURE_COLUMNS = [
    'home_off_form', 'away_off_form',
    'home_def_form', 'away_def_form',
    'home_strength', 'away_strength',
    'home_matches_played', 'away_matches_played',
    'strength_diff', 'off_form_diff', 'def_form_diff',
    'experience_ratio',
    'home_team_tier', 'away_team_tier',
    'home_fifa_points', 'away_fifa_points', 'fifa_points_diff',
    'neutral'
]


def calculate_poisson_probabilities(home_xg: float, away_xg: float, max_goals: int = 10) -> dict:
    prob_home_win = prob_draw = prob_away_win = 0.0
    for i in range(max_goals):
        for j in range(max_goals):
            p = poisson.pmf(i, home_xg) * poisson.pmf(j, away_xg)
            if i > j:
                prob_home_win += p
            elif i == j:
                prob_draw += p
            else:
                prob_away_win += p
    return {'home_win': prob_home_win * 100, 'draw': prob_draw * 100, 'away_win': prob_away_win * 100}


def main():
    try:
        logging.info("1. Cargando dataset con features...")
        df = pd.read_csv(FEATURES_DATA_PATH)
        df['date'] = pd.to_datetime(df['date'])
        df = df[df['date'].dt.year >= 1990].copy()
        logging.info(f"   -> Filtrado a era moderna: {len(df)} partidos.")

        logging.info("2. Preparando datos para regresión...")
        X = df[FEATURE_COLUMNS]
        y_home = df['home_score']
        y_away = df['away_score']

        logging.info("3. Dividiendo en Train (80%) y Test (20%)...")
        X_train, X_test, y_home_train, y_home_test, y_away_train, y_away_test = train_test_split(
            X, y_home, y_away, test_size=0.2, random_state=42
        )

        logging.info("4. Entrenando modelo de Goles del LOCAL...")
        model_home = XGBRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            min_child_weight=3, subsample=0.8, colsample_bytree=0.8, random_state=42
        )
        model_home.fit(X_train, y_home_train)
        logging.info("   -> ¡Modelo local entrenado!")

        logging.info("5. Entrenando modelo de Goles del VISITANTE...")
        model_away = XGBRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            min_child_weight=3, subsample=0.8, colsample_bytree=0.8, random_state=42
        )
        model_away.fit(X_train, y_away_train)
        logging.info("   -> ¡Modelo visitante entrenado!")

        logging.info("6. Evaluando modelos...")
        home_pred = model_home.predict(X_test)
        away_pred = model_away.predict(X_test)
        logging.info(f"   -> Modelo LOCAL - MAE: {mean_absolute_error(y_home_test, home_pred):.3f} goles")
        logging.info(f"   -> Modelo VISITANTE - MAE: {mean_absolute_error(y_away_test, away_pred):.3f} goles")

        logging.info("7. Guardando modelos...")
        joblib.dump(model_home, MODEL_HOME_PATH)
        joblib.dump(model_away, MODEL_AWAY_PATH)
        logging.info(f"   -> ✅ Modelos guardados en: {PROJECT_DIR / 'src' / 'models'}")

    except Exception as e:
        logging.error(f"Error durante el entrenamiento: {e}")


if __name__ == "__main__":
    main()