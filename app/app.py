# app/app.py
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Configuración de la página
st.set_page_config(page_title="ScoutData AI", page_icon="⚽", layout="centered")

# Rutas
PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "src" / "models" / "xgboost_model.joblib"
DATA_PATH = PROJECT_DIR / "data" / "processed" / "matches_with_features.csv"

# Cargar modelo y datos (con caché para que sea rápido)
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    # Filtramos a era moderna para obtener equipos relevantes
    return df[df['date'].dt.year >= 2010]

model = load_model()
df = load_data()

# Obtener lista única de equipos para los menús
teams = sorted(df['home_team'].unique())

# --- INTERFAZ DE USUARIO ---
st.title("⚽ ScoutData: Predicción de Partidos")
st.markdown("Modelo de Machine Learning (XGBoost) entrenado con datos históricos de fútbol internacional.")

st.divider()

# Entradas del usuario
col1, col2 = st.columns(2)

with col1:
    home_team = st.selectbox("🏠 Equipo Local", teams, index=teams.index("Argentina") if "Argentina" in teams else 0)
    home_form = st.number_input("Forma reciente Local (Promedio goles últimos 5 partidos)", min_value=0.0, max_value=5.0, value=1.5, step=0.1)

with col2:
    away_team = st.selectbox("✈️ Equipo Visitante", teams, index=teams.index("Brasil") if "Brasil" in teams else 1)
    away_form = st.number_input("Forma reciente Visitante (Promedio goles últimos 5 partidos)", min_value=0.0, max_value=5.0, value=1.0, step=0.1)

neutral = st.checkbox("¿El partido es en cancha neutral? (Ej: Final de torneo)", value=False)

# Botón de predicción
if st.button("🔮 Predecir Resultado", type="primary", use_container_width=True):
    with st.spinner("Analizando datos históricos y forma reciente..."):
        # Preparar los datos exactamente como los vio el modelo en entrenamiento
        input_data = pd.DataFrame({
            'home_team_form': [home_form],
            'away_team_form': [away_form],
            'neutral': [1 if neutral else 0]
        })
        
        # Asegurar el orden de las columnas (buena práctica)
        input_data = input_data[['home_team_form', 'away_team_form', 'neutral']]
        
        # Predecir probabilidad (clase 1 = Gana Local)
        prob_local_win = model.predict_proba(input_data)[0][1] * 100
        
        # Mostrar resultados
        st.divider()
        st.subheader(f"Resultado para: {home_team} vs {away_team}")
        
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Probabilidad Victoria Local", f"{prob_local_win:.1f}%")
        col_b.metric("Probabilidad No Victoria Local", f"{100 - prob_local_win:.1f}%")
        
        # Mensaje contextual
        if prob_local_win > 65:
            st.success(f"🟢 El modelo favorece claramente a **{home_team}** como local.")
        elif prob_local_win < 35:
            st.error(f"🔴 El modelo indica que **{home_team}** tiene bajas probabilidades de ganar. Favorito: **{away_team}** o Empate.")
        else:
            st.warning("🟡 Partido muy parejo. El modelo no encuentra un claro favorito.")