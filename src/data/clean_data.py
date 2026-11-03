# src/data/clean_data.py
import pandas as pd
import logging
from pathlib import Path

# Configuración del Logger (para ver qué está pasando en la consola)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Rutas automáticas (funciona en Windows, Mac y Linux)
PROJECT_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_DIR / "data" / "processed"


def load_and_clean_data(input_file: str) -> pd.DataFrame:
    """Carga y limpia el dataset de partidos internacionales."""
    input_path = RAW_DATA_DIR / input_file
    logging.info(f"Cargando datos desde: {input_path}")
    
    # 1. Cargar datos
    df = pd.read_csv(input_path)
    initial_rows = len(df)
    logging.info(f"Dataset cargado: {initial_rows} partidos.")
    
    # 2. Formatear fechas
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # 3. Eliminar filas con datos críticos faltantes
    cols_to_check = ['date', 'home_team', 'away_team', 'home_score', 'away_score']
    df = df.dropna(subset=cols_to_check)
    
    # 4. Feature Engineering (¡Tu primera victoria de datos!)
    df['goal_difference'] = df['home_score'] - df['away_score']
    df['total_goals'] = df['home_score'] + df['away_score']
    
    # Crear variables objetivo binarias para clasificación (1 = Sí, 0 = No)
    df['home_win'] = (df['goal_difference'] > 0).astype(int)
    df['draw'] = (df['goal_difference'] == 0).astype(int)
    df['away_win'] = (df['goal_difference'] < 0).astype(int)
    
    # 5. Ordenar cronológicamente (crucial para series de tiempo en fútbol)
    df = df.sort_values('date').reset_index(drop=True)
    
    logging.info(f"Limpieza completada. Filas finales: {len(df)}")
    return df


def save_clean_data(df: pd.DataFrame, output_file: str) -> None:
    """Guarda el dataset procesado en la carpeta correspondiente."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DATA_DIR / output_file
    df.to_csv(output_path, index=False)
    logging.info(f"✅ Dataset guardado exitosamente en: {output_path}")


if __name__ == "__main__":
    DATASET_NAME = "results.csv"
    
    try:
        clean_df = load_and_clean_data(DATASET_NAME)
        save_clean_data(clean_df, "matches_clean.csv")
    except FileNotFoundError:
        logging.error(f"❌ No se encontró '{DATASET_NAME}' en la carpeta data/raw/")
        logging.error("💡 Asegurate de haber descargado y descomprimido el CSV de Kaggle ahí.")
    except Exception as e:
        logging.error(f"❌ Error inesperado: {e}")