# src/data/create_features.py
import pandas as pd
import logging
from pathlib import Path

# Configuración de Logs
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Rutas
PROJECT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DATA_PATH = PROJECT_DIR / "data" / "processed" / "matches_clean.csv"
FEATURES_DATA_PATH = PROJECT_DIR / "data" / "processed" / "matches_with_features.csv"

def calculate_team_form(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Calcula la 'forma' (promedio de goles anotados) de cada equipo 
    en sus últimos 'window' partidos ANTES del partido actual.
    Evita Data Leakage al actualizar el historial después de calcular.
    """
    logging.info(f"Calculando forma de equipos (ventana = {window} partidos)...")
    
    # Inicializamos las nuevas columnas
    df['home_team_form'] = 0.0
    df['away_team_form'] = 0.0
    
    # Diccionario para guardar el historial de cada equipo
    # Estructura: {'Argentina': [{'goals': 2}, {'goals': 1}, ...], 'Brasil': [...]}
    team_history = {}
    
    # Iteramos sobre el dataframe (ya está ordenado por fecha gracias a clean_data.py)
    for index, row in df.iterrows():
        home_team = row['home_team']
        away_team = row['away_team']
        
        # 1. Calcular forma del equipo LOCAL
        home_matches = team_history.get(home_team, [])
        # Tomamos los últimos 'window' partidos. Si hay menos, usamos los que haya (min_periods)
        recent_home = home_matches[-window:]
        avg_home_goals = sum(m['goals_scored'] for m in recent_home) / max(len(recent_home), 1)
        df.at[index, 'home_team_form'] = avg_home_goals
        
        # 2. Calcular forma del equipo VISITANTE
        away_matches = team_history.get(away_team, [])
        recent_away = away_matches[-window:]
        avg_away_goals = sum(m['goals_scored'] for m in recent_away) / max(len(recent_away), 1)
        df.at[index, 'away_team_form'] = avg_away_goals
        
        # 3. ACTUALIZAR el historial (¡Después de calcular! Esto evita Data Leakage)
        if home_team not in team_history:
            team_history[home_team] = []
        team_history[home_team].append({'goals_scored': row['home_score']})
        
        if away_team not in team_history:
            team_history[away_team] = []
        team_history[away_team].append({'goals_scored': row['away_score']})
        
    logging.info("✅ Cálculo de formas completado.")
    return df

def main():
    try:
        logging.info(f"Cargando datos desde: {PROCESSED_DATA_PATH}")
        df = pd.read_csv(PROCESSED_DATA_PATH)
        df['date'] = pd.to_datetime(df['date'])
        
        # Aplicar Feature Engineering
        df_featured = calculate_team_form(df, window=5)
        
        # Guardar el nuevo dataset
        df_featured.to_csv(FEATURES_DATA_PATH, index=False)
        logging.info(f"🚀 Dataset con features guardado en: {FEATURES_DATA_PATH}")
        logging.info(f"📊 Nuevas columnas creadas: {['home_team_form', 'away_team_form']}")
        
    except Exception as e:
        logging.error(f"Error en el pipeline de features: {e}")

if __name__ == "__main__":
    main()