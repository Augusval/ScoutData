# src/models/train_model.py
import pandas as pd
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import joblib

# Configuración de Logs
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Rutas
PROJECT_DIR = Path(__file__).resolve().parents[2]
FEATURES_DATA_PATH = PROJECT_DIR / "data" / "processed" / "matches_with_features.csv"
MODEL_PATH = PROJECT_DIR / "src" / "models" / "xgboost_model.joblib"

def main():
    try:
        logging.info("1. Cargando dataset con features...")
        df = pd.read_csv(FEATURES_DATA_PATH)
        
        # Filtramos solo partidos desde 1990 para que el modelo aprenda patrones del fútbol moderno
        # (El fútbol de 1870 es demasiado diferente al actual)
        df['date'] = pd.to_datetime(df['date'])
        df = df[df['date'].dt.year >= 1990].copy()
        logging.info(f"   -> Filtrado a era moderna: {len(df)} partidos.")

        logging.info("2. Definiendo Features (X) y Target (y)...")
        # Features: Variables que el modelo usará para predecir (SOLO lo que sabemos ANTES del partido)
        features_to_use = [
            'home_team_form', 
            'away_team_form',
            'neutral' # 1 si es cancha neutral (ej: final de mundial), 0 si no
        ]
        
        # Target: Lo que queremos predecir (1 = Gana Local, 0 = No gana local)
        target = 'home_win'
        
        X = df[features_to_use]
        y = df[target]

        logging.info("3. Dividiendo en Train (80%) y Test (20%)...")
        # random_state=42 asegura que la división sea reproducible
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        logging.info("4. Entrenando modelo XGBoost...")
        # Inicializamos el clasificador con hiperparámetros sólidos para empezar
        model = XGBClassifier(
            n_estimators=100,      # Número de árboles
            max_depth=4,           # Profundidad máxima (evita overfitting)
            learning_rate=0.1,     # Tasa de aprendizaje
            random_state=42,
            eval_metric='logloss'
        )
        
        model.fit(X_train, y_train)
        logging.info("   -> ¡Entrenamiento completado!")

        logging.info("5. Evaluando el modelo...")
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        logging.info(f"   -> Precisión (Accuracy) en datos de Test: {accuracy:.4f} ({accuracy*100:.2f}%)")
        logging.info("\nReporte de Clasificación:\n" + classification_report(y_test, y_pred, target_names=['No Gana Local (0)', 'Gana Local (1)']))

        logging.info("6. Guardando el modelo entrenado...")
        joblib.dump(model, MODEL_PATH)
        logging.info(f"   -> ✅ Modelo guardado exitosamente en: {MODEL_PATH}")

    except Exception as e:
        logging.error(f"Error durante el entrenamiento: {e}")

if __name__ == "__main__":
    main()