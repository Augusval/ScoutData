"""
ScoutData AI V10 - Hybrid Ensemble (XGBoost + Parametric Elo Calibration)
==========================================================================
Dashboard Streamlit que combina DOS señales para predecir goles esperados:
  1. XGBoost Regressors (forma ofensiva/defensiva, fuerza histórica, features)
  2. Capa paramétrica basada en Rating FIFA (extrapola a matchups extremos)

El blend usa gating por magnitud del matchup: en partidos parejos domina el
modelo ML; en desbalanceados entra la fórmula paramétrica. Las probabilidades
finales se calculan con Distribución de Poisson.

Autor: Augusto Valentini
Stack: Python, Pandas, XGBoost, Scikit-Learn, Streamlit, SciPy, BeautifulSoup
"""
import streamlit as st
import pandas as pd
import joblib
import logging
import sys
from pathlib import Path
from datetime import datetime
import math
from scipy.stats import poisson

# Agregar src/data al path para importar el módulo de FIFA
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src" / "data"))
from fetch_fifa_ranking import get_fifa_ranking, get_team_fifa

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

st.set_page_config(page_title="ScoutData AI V10", page_icon="⚽", layout="wide")

MODEL_HOME_PATH = PROJECT_DIR / "src" / "models" / "xgboost_home_goals.joblib"
MODEL_AWAY_PATH = PROJECT_DIR / "src" / "models" / "xgboost_away_goals.joblib"
DATA_PATH = PROJECT_DIR / "data" / "processed" / "matches_with_features.csv"

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

FORM_WINDOW = 5

# Constantes de la capa paramétrica (calibradas al fútbol internacional)
BASE_HOME_GOALS = 1.45     # goles promedio de un local en intl football
BASE_AWAY_GOALS = 1.10     # goles promedio de un visitante en intl football
NEUTRAL_BASE = 1.28        # sin ventaja de local en cancha neutral
ELO_SENSITIVITY = 0.25     # sensibilidad goles / rating (escala Elo)
BLEND_SATURATION = 3.0     # adv=3.0 (diff ~1200) -> 100% paramétrico


# ---------------------------------------------------------------------------
# CARGA
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_models():
    return joblib.load(MODEL_HOME_PATH), joblib.load(MODEL_AWAY_PATH)


@st.cache_data(show_spinner=False, ttl=3600)
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"].dt.year >= 2010].copy()
    return df.sort_values("date", ascending=False).reset_index(drop=True)


@st.cache_data(show_spinner=False, ttl=86400)
def load_fifa_ranking() -> dict:
    return get_fifa_ranking()


# ---------------------------------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------------------------------

def get_team_stats(team_name: str, df: pd.DataFrame, window: int = FORM_WINDOW):
    matches = df[(df['home_team'] == team_name) | (df['away_team'] == team_name)]
    if matches.empty:
        return {'off': 0.0, 'def': 0.0, 'strength': 0.4, 'matches': 0, 'wins': 0, 'tier': 0.4}

    recent = matches.head(window)
    gs, gc = [], []
    for _, row in recent.iterrows():
        if row['home_team'] == team_name:
            gs.append(row['home_score']); gc.append(row['away_score'])
        else:
            gs.append(row['away_score']); gc.append(row['home_score'])

    off_form = sum(gs) / len(gs)
    def_form = sum(gc) / len(gc)

    total = len(matches)
    wins = 0
    total_gs = 0
    for _, row in matches.iterrows():
        if row['home_team'] == team_name:
            total_gs += row['home_score']
            if row['home_score'] > row['away_score']:
                wins += 1
        else:
            total_gs += row['away_score']
            if row['away_score'] > row['home_score']:
                wins += 1

    GLOBAL_WIN_RATE = df['home_win'].mean()
    K = 200
    strength = (wins + GLOBAL_WIN_RATE * K) / (total + K)
    avg_goals = total_gs / total
    tier = (strength * 0.5) + (avg_goals * 0.3) + (math.log(total + 1) * 0.02)

    return {'off': off_form, 'def': def_form, 'strength': strength,
            'matches': total, 'wins': wins, 'tier': tier}


# ---------------------------------------------------------------------------
# CAPA PARAMÉTRICA (Hybrid Ensemble)
# ---------------------------------------------------------------------------

def parametric_xg(fifa_home: float, fifa_away: float, neutral: bool) -> tuple:
    """
    Calcula goles esperados SOLO desde ratings FIFA (extrapola a extremos).
    Fórmula exponencial calibrada al promedio de goles del fútbol internacional.
    """
    if neutral:
        base_home, base_away = NEUTRAL_BASE, NEUTRAL_BASE
    else:
        base_home, base_away = BASE_HOME_GOALS, BASE_AWAY_GOALS

    advantage = (fifa_home - fifa_away) / 400.0  # escala Elo estándar
    lam_home = base_home * math.exp(ELO_SENSITIVITY * advantage)
    lam_away = base_away * math.exp(-ELO_SENSITIVITY * advantage)
    return max(0.05, lam_home), max(0.05, lam_away)


def blend_weight(fifa_home: float, fifa_away: float) -> float:
    """
    Peso de la capa paramétrica: crece con lo EXTREMO del matchup.
    Partidos parejos -> modelo ML domina. Desbalanceados -> fórmula entra.
    """
    advantage = abs(fifa_home - fifa_away) / 400.0
    return min(1.0, max(0.0, advantage / BLEND_SATURATION))


def calculate_poisson_probabilities(home_xg: float, away_xg: float, max_goals: int = 10) -> dict:
    ph = pd_ = pa = 0.0
    for i in range(max_goals):
        for j in range(max_goals):
            p = poisson.pmf(i, home_xg) * poisson.pmf(j, away_xg)
            if i > j: ph += p
            elif i == j: pd_ += p
            else: pa += p
    return {'home_win': ph * 100, 'draw': pd_ * 100, 'away_win': pa * 100}


def obtener_ultimos_partidos(team_name: str, df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    team_matches = df[(df['home_team'] == team_name) | (df['away_team'] == team_name)].head(n)
    resultados = []
    for _, row in team_matches.iterrows():
        if row['home_team'] == team_name:
            goles, rival, local = row['home_score'], row['away_team'], True
        else:
            goles, rival, local = row['away_score'], row['home_team'], False
        if row['home_score'] == row['away_score']:
            res = "E"
        elif (local and row['home_score'] > row['away_score']) or (not local and row['away_score'] > row['home_score']):
            res = "G"
        else:
            res = "P"
        resultados.append({
            "Fecha": row["date"].strftime("%Y-%m-%d"), "Local": row["home_team"],
            "Visitante": row["away_team"], "Goles": goles, "Rival": rival, "Resultado": res,
        })
    return pd.DataFrame(resultados)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def main():
    model_home, model_away = load_models()
    df = load_data()
    fifa_ranking = load_fifa_ranking()
    teams = sorted(df["home_team"].unique())
    default_home = teams.index("France") if "France" in teams else 0

    with st.sidebar:
        st.image("https://img.icons8.com/color/96/football2.png", width=80)
        st.title("ScoutData AI V10")
        st.caption("Hybrid Ensemble: XGBoost + Elo Calibration")
        st.divider()
        st.markdown("###  Arquitectura del modelo")
        st.markdown("- **Capa 1:** XGBoost Regressor (x2)")
        st.markdown("- **Capa 2:** Paramétrica (Rating FIFA → λ)")
        st.markdown("- **Blend:** gating por magnitud del matchup")
        st.markdown("- **Probabilidades:** Poisson")
        st.markdown(f"- **Features ML:** {len(FEATURE_COLUMNS)}")
        st.markdown(f"- **Ranking FIFA:** {len(fifa_ranking)} equipos")
        st.divider()
        st.markdown("### 🛠️ Stack")
        st.markdown("- Python, Pandas, XGBoost")
        st.markdown("- SciPy (Poisson)")
        st.markdown("- Streamlit, BeautifulSoup")
        st.divider()
        st.markdown(f"⚽ *Desarrollado por Augusto Valentini*")

    st.title("⚽ ScoutData AI V10: Hybrid Ensemble")
    st.markdown(
        "Predicción que combina un **modelo de Machine Learning (XGBoost)** con una "
        "**capa paramétrica de calibración basada en Rating FIFA**. El blend se ajusta "
        "automáticamente: en partidos parejos domina el modelo; en desbalanceados entra "
        "la fórmula, que extrapola correctamente a matchups extremos."
    )
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🏠 Equipo Local")
        home_team = st.selectbox("Seleccioná el equipo local", teams, index=default_home, key="home")
    with col2:
        st.subheader("✈️ Equipo Visitante")
        available = [t for t in teams if t != home_team]
        away_idx = available.index("Arameans Suryoye") if "Arameans Suryoye" in available else 0
        away_team = st.selectbox("Seleccioná el equipo visitante", available, index=away_idx, key="away")

    neutral = st.checkbox("🏟️ ¿Cancha neutral?", value=False)

    home_stats = get_team_stats(home_team, df)
    away_stats = get_team_stats(away_team, df)
    home_fifa = get_team_fifa(home_team, fifa_ranking)
    away_fifa = get_team_fifa(away_team, fifa_ranking)

    strength_diff = home_stats['strength'] - away_stats['strength']
    off_form_diff = home_stats['off'] - away_stats['off']
    def_form_diff = away_stats['def'] - home_stats['def']
    experience_ratio = math.log(home_stats['matches'] / away_stats['matches']) if home_stats['matches'] > 0 and away_stats['matches'] > 0 else 0.0
    fifa_diff = home_fifa['points'] - away_fifa['points']

    # Peso del blend (se muestra para transparencia)
    w_param = blend_weight(home_fifa['points'], away_fifa['points'])

    def fmt_rank(info):
        return "Sin ranking FIFA (amateur)" if info['rank'] is None else f"#{info['rank']} ({info['points']:.0f} pts)"

    st.info(
        f"📊 **Análisis de Equipo:**\n\n"
        f"**{home_team}** (Local) - {home_stats['matches']} partidos | 🌍 FIFA: {fmt_rank(home_fifa)}\n"
        f"- ⚽ Ataque: {home_stats['off']:.2f} | 🛡️ Defensa: {home_stats['def']:.2f} | 💪 Fuerza: {home_stats['strength']*100:.1f}%\n\n"
        f"**{away_team}** (Visitante) - {away_stats['matches']} partidos | 🌍 FIFA: {fmt_rank(away_fifa)}\n"
        f"- ⚽ Ataque: {away_stats['off']:.2f} | 🛡️ Defensa: {away_stats['def']:.2f} | 💪 Fuerza: {away_stats['strength']*100:.1f}%\n\n"
        f"**📈 Diferencia FIFA:** {fifa_diff:+.0f} puntos | "
        f"**⚖️ Peso calibración paramétrica:** {w_param*100:.0f}% "
        f"(modelo ML: {(1-w_param)*100:.0f}%)"
    )

    if st.button("🔮 Predecir (Hybrid Ensemble)", type="primary", use_container_width=True):
        with st.spinner("Combinando modelo ML + calibración paramétrica..."):
            # --- Capa 1: XGBoost ---
            input_data = pd.DataFrame([{
                'home_off_form': home_stats['off'], 'away_off_form': away_stats['off'],
                'home_def_form': home_stats['def'], 'away_def_form': away_stats['def'],
                'home_strength': home_stats['strength'], 'away_strength': away_stats['strength'],
                'home_matches_played': home_stats['matches'], 'away_matches_played': away_stats['matches'],
                'strength_diff': strength_diff, 'off_form_diff': off_form_diff,
                'def_form_diff': def_form_diff, 'experience_ratio': experience_ratio,
                'home_team_tier': home_stats['tier'], 'away_team_tier': away_stats['tier'],
                'home_fifa_points': home_fifa['points'], 'away_fifa_points': away_fifa['points'],
                'fifa_points_diff': fifa_diff,
                'neutral': 1 if neutral else 0
            }])[FEATURE_COLUMNS]

            xg_home_ml = max(0.0, float(model_home.predict(input_data)[0]))
            xg_away_ml = max(0.0, float(model_away.predict(input_data)[0]))

            # --- Capa 2: Paramétrica (Elo/FIFA) ---
            xg_home_par, xg_away_par = parametric_xg(home_fifa['points'], away_fifa['points'], neutral)

            # --- Blend con gating ---
            home_xg = (1 - w_param) * xg_home_ml + w_param * xg_home_par
            away_xg = (1 - w_param) * xg_away_ml + w_param * xg_away_par

            probs = calculate_poisson_probabilities(home_xg, away_xg)

            st.divider()
            st.subheader(f"📈 Predicción: {home_team} vs {away_team}")

            # Desglose del blend (transparencia total)
            st.markdown("### 🧬 Desglose del Hybrid Ensemble")
            bd1, bd2, bd3 = st.columns(3)
            bd1.metric("xG Modelo ML", f"{xg_home_ml:.2f} - {xg_away_ml:.2f}")
            bd2.metric("xG Paramétrico (FIFA)", f"{xg_home_par:.2f} - {xg_away_par:.2f}")
            bd3.metric("Blend aplicado", f"{w_param*100:.0f}% param. / {(1-w_param)*100:.0f}% ML")

            st.markdown("### ⚽ Goles Esperados (xG final)")
            c1, c2 = st.columns(2)
            c1.metric(f"Goles esperados {home_team}", f"{home_xg:.2f}")
            c2.metric(f"Goles esperados {away_team}", f"{away_xg:.2f}")
            st.info(f"🎯 **Marcador más probable:** {home_team} {round(home_xg)} - {round(away_xg)} {away_team}")

            st.divider()
            st.markdown("### 📊 Probabilidades (Poisson)")
            m1, m2, m3 = st.columns(3)
            m1.metric(f"Victoria {home_team}", f"{probs['home_win']:.1f}%")
            m2.metric("Empate", f"{probs['draw']:.1f}%")
            m3.metric(f"Victoria {away_team}", f"{probs['away_win']:.1f}%")

            if probs['home_win'] > 75:
                st.success(f"🟢 **{home_team}** es el favorito abrumador ({probs['home_win']:.1f}%).")
            elif probs['home_win'] > 60:
                st.info(f"🔵 **{home_team}** es el claro favorito.")
            elif probs['away_win'] > 75:
                st.error(f"🔴 **{away_team}** es el favorito abrumador ({probs['away_win']:.1f}%).")
            elif probs['away_win'] > 60:
                st.warning(f"🟠 **{away_team}** es el favorito.")
            else:
                st.warning("🟡 Partido parejo. Cualquiera puede ganar.")

            st.divider()
            st.subheader("📋 Últimos 5 partidos")
            ch1, ch2 = st.columns(2)
            with ch1:
                st.markdown(f"**{home_team}**")
                h = obtener_ultimos_partidos(home_team, df)
                if not h.empty:
                    st.dataframe(h, use_container_width=True, hide_index=True)
            with ch2:
                st.markdown(f"**{away_team}**")
                a = obtener_ultimos_partidos(away_team, df)
                if not a.empty:
                    st.dataframe(a, use_container_width=True, hide_index=True)

            if "historial" not in st.session_state:
                st.session_state.historial = []
            st.session_state.historial.append({
                "Hora": datetime.now().strftime("%H:%M:%S"), "Local": home_team,
                "Visitante": away_team, "xG Local": f"{home_xg:.2f}",
                "xG Visitante": f"{away_xg:.2f}", "Prob. Local": f"{probs['home_win']:.1f}%",
                "Blend param.": f"{w_param*100:.0f}%",
            })

    if "historial" in st.session_state and st.session_state.historial:
        st.divider()
        st.subheader("🕒 Historial de la sesión")
        st.dataframe(pd.DataFrame(st.session_state.historial), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()