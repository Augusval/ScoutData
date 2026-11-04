# src/data/create_features.py
import pandas as pd
import logging
from pathlib import Path
import math

from fetch_fifa_ranking import get_fifa_ranking, get_team_fifa

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

PROJECT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DATA_PATH = PROJECT_DIR / "data" / "processed" / "matches_clean.csv"
FEATURES_DATA_PATH = PROJECT_DIR / "data" / "processed" / "matches_with_features.csv"


def calculate_advanced_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Calcula features avanzadas evitando Data Leakage.
    V9: Incluye Team Tier y puntos del Ranking FIFA como features de nivel absoluto.
    """
    logging.info("Calculando features avanzadas (V9 con Team Tier + Ranking FIFA)...")

    fifa_ranking = get_fifa_ranking()

    df = df.copy()
    for col in ['home_off_form', 'away_off_form', 'home_def_form', 'away_def_form',
                'home_strength', 'away_strength', 'strength_diff', 'off_form_diff',
                'def_form_diff', 'experience_ratio', 'home_team_tier', 'away_team_tier',
                'home_fifa_points', 'away_fifa_points', 'fifa_points_diff']:
        df[col] = 0.0
    df['home_matches_played'] = 0
    df['away_matches_played'] = 0

    GLOBAL_WIN_RATE = df['home_win'].mean()
    K = 200

    team_history = {}

    for index, row in df.iterrows():
        home_team = row['home_team']
        away_team = row['away_team']
        home_goals = row['home_score']
        away_goals = row['away_score']

        for team in [home_team, away_team]:
            if team not in team_history:
                team_history[team] = {
                    'wins': 0, 'matches': 0,
                    'goals_scored': [], 'goals_conceded': [],
                    'total_goals_scored': 0
                }

        home_hist = team_history[home_team]
        away_hist = team_history[away_team]

        # Forma reciente (últimos 'window' partidos)
        rh_s = home_hist['goals_scored'][-window:]
        ra_s = away_hist['goals_scored'][-window:]
        rh_c = home_hist['goals_conceded'][-window:]
        ra_c = away_hist['goals_conceded'][-window:]

        df.loc[index, 'home_off_form'] = sum(rh_s) / max(len(rh_s), 1)
        df.loc[index, 'away_off_form'] = sum(ra_s) / max(len(ra_s), 1)
        df.loc[index, 'home_def_form'] = sum(rh_c) / max(len(rh_c), 1)
        df.loc[index, 'away_def_form'] = sum(ra_c) / max(len(ra_c), 1)

        # Fuerza con regularización bayesiana (K=200)
        home_strength = (home_hist['wins'] + GLOBAL_WIN_RATE * K) / (home_hist['matches'] + K)
        away_strength = (away_hist['wins'] + GLOBAL_WIN_RATE * K) / (away_hist['matches'] + K)
        df.loc[index, 'home_strength'] = home_strength
        df.loc[index, 'away_strength'] = away_strength

        df.loc[index, 'home_matches_played'] = home_hist['matches']
        df.loc[index, 'away_matches_played'] = away_hist['matches']

        df.loc[index, 'strength_diff'] = home_strength - away_strength
        df.loc[index, 'off_form_diff'] = df.loc[index, 'home_off_form'] - df.loc[index, 'away_off_form']
        df.loc[index, 'def_form_diff'] = df.loc[index, 'away_def_form'] - df.loc[index, 'home_def_form']

        if home_hist['matches'] > 0 and away_hist['matches'] > 0:
            df.loc[index, 'experience_ratio'] = math.log(home_hist['matches'] / away_hist['matches'])

        # Team Tier (nivel absoluto basado en historial propio)
        if home_hist['matches'] > 0:
            h_avg = home_hist['total_goals_scored'] / home_hist['matches']
            df.loc[index, 'home_team_tier'] = (home_strength * 0.5) + (h_avg * 0.3) + (math.log(home_hist['matches'] + 1) * 0.02)
        if away_hist['matches'] > 0:
            a_avg = away_hist['total_goals_scored'] / away_hist['matches']
            df.loc[index, 'away_team_tier'] = (away_strength * 0.5) + (a_avg * 0.3) + (math.log(away_hist['matches'] + 1) * 0.02)

        # Ranking FIFA (nivel absoluto oficial)
        home_fifa = get_team_fifa(home_team, fifa_ranking)['points']
        away_fifa = get_team_fifa(away_team, fifa_ranking)['points']
        df.loc[index, 'home_fifa_points'] = home_fifa
        df.loc[index, 'away_fifa_points'] = away_fifa
        df.loc[index, 'fifa_points_diff'] = home_fifa - away_fifa

        # Actualizar historial DESPUÉS de calcular (evita Data Leakage)
        home_hist['matches'] += 1
        away_hist['matches'] += 1
        if home_goals > away_goals:
            home_hist['wins'] += 1
        elif away_goals > home_goals:
            away_hist['wins'] += 1
        home_hist['goals_scored'].append(home_goals)
        home_hist['goals_conceded'].append(away_goals)
        home_hist['total_goals_scored'] += home_goals
        away_hist['goals_scored'].append(away_goals)
        away_hist['goals_conceded'].append(home_goals)
        away_hist['total_goals_scored'] += away_goals

    logging.info("✅ Cálculo de features V9 completado.")
    return df


def main():
    try:
        logging.info("Cargando datos limpios...")
        df = pd.read_csv(PROCESSED_DATA_PATH)
        df['date'] = pd.to_datetime(df['date'])

        df_featured = calculate_advanced_features(df, window=5)
        df_featured.to_csv(FEATURES_DATA_PATH, index=False)
        logging.info(f"🚀 Dataset V9 guardado en: {FEATURES_DATA_PATH}")

    except Exception as e:
        logging.error(f"Error: {e}")


if __name__ == "__main__":
    main()