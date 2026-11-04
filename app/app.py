"""
ScoutData AI V11 - Football Analytics Hub (estilo FotMob / SofaScore)
======================================================================
App Streamlit multi-página (router por session_state, single-file):
  ⚽ Partidos   -> jornada sugerida con predicciones inline (fixture hub)
  🔮 Predicción -> detalle de un matchup con desglose del hybrid ensemble
  🏆 Ranking    -> tabla FIFA/ELO navegable con barras de progreso
  👥 Equipos    -> ficha de equipo + rivales sugeridos
  ℹ️ Modelo     -> arquitectura, stack, limitaciones y autor

Motor: Hybrid Ensemble (dual XGBoost + capa paramétrica Elo/FIFA) con
gating por magnitud del matchup, y probabilidades vía Poisson.

Autor: Augusto Valentini
Stack: Python, Pandas, XGBoost, Scikit-Learn, Streamlit, SciPy, BeautifulSoup
"""
import streamlit as st
import pandas as pd
import joblib
import logging
import sys
import unicodedata
from pathlib import Path
from datetime import datetime
import math
from scipy.stats import poisson

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src" / "data"))
from fetch_fifa_ranking import get_fifa_ranking, get_team_fifa

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

st.set_page_config(page_title="ScoutData — Football Analytics Hub", page_icon="⚽", layout="wide")

MODEL_HOME_PATH = PROJECT_DIR / "src" / "models" / "xgboost_home_goals.joblib"
MODEL_AWAY_PATH = PROJECT_DIR / "src" / "models" / "xgboost_away_goals.joblib"
DATA_PATH = PROJECT_DIR / "data" / "processed" / "matches_with_features.csv"

FEATURE_COLUMNS = [
    'home_off_form', 'away_off_form', 'home_def_form', 'away_def_form',
    'home_strength', 'away_strength', 'home_matches_played', 'away_matches_played',
    'strength_diff', 'off_form_diff', 'def_form_diff', 'experience_ratio',
    'home_team_tier', 'away_team_tier',
    'home_fifa_points', 'away_fifa_points', 'fifa_points_diff', 'neutral'
]
FORM_WINDOW = 5

# Constantes de la capa paramétrica (calibradas al fútbol internacional)
BASE_HOME_GOALS, BASE_AWAY_GOALS, NEUTRAL_BASE = 1.45, 1.10, 1.28
ELO_SENSITIVITY, BLEND_SATURATION = 0.25, 3.0

# Jornada sugerida (matchups curados; se filtran por existencia en el dataset)
FIXTURES = [
    ("Spain", "France"), ("Argentina", "Brazil"), ("England", "Germany"),
    ("Portugal", "Netherlands"), ("Italy", "Belgium"), ("Morocco", "Japan"),
    ("France", "Arameans Suryoye"), ("Spain", "Afghanistan"), ("Argentina", "Arameans Suryoye"),
]


# ---------------------------------------------------------------------------
# CARGA (caché)
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
# NORMALIZACIÓN (para casar nombres del ranking con nombres del dataset)
# ---------------------------------------------------------------------------
def _norm(s: str) -> str:
    s = unicodedata.normalize('NFKD', str(s))
    s = ''.join(c for c in s if not unicodedata.combining(c))
    return s.lower().replace('.', '').replace('-', ' ').strip()


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
    total = len(matches); wins = 0; total_gs = 0
    for _, row in matches.iterrows():
        if row['home_team'] == team_name:
            total_gs += row['home_score']
            if row['home_score'] > row['away_score']: wins += 1
        else:
            total_gs += row['away_score']
            if row['away_score'] > row['home_score']: wins += 1
    GLOBAL_WIN_RATE = df['home_win'].mean(); K = 200
    strength = (wins + GLOBAL_WIN_RATE * K) / (total + K)
    avg_goals = total_gs / total
    tier = (strength * 0.5) + (avg_goals * 0.3) + (math.log(total + 1) * 0.02)
    return {'off': off_form, 'def': def_form, 'strength': strength,
            'matches': total, 'wins': wins, 'tier': tier}


# ---------------------------------------------------------------------------
# MOTOR: HYBRID ENSEMBLE (una sola fuente de verdad, reusada por todas las páginas)
# ---------------------------------------------------------------------------
def parametric_xg(fifa_home: float, fifa_away: float, neutral: bool) -> tuple:
    bh, ba = (NEUTRAL_BASE, NEUTRAL_BASE) if neutral else (BASE_HOME_GOALS, BASE_AWAY_GOALS)
    adv = (fifa_home - fifa_away) / 400.0
    return max(0.05, bh * math.exp(ELO_SENSITIVITY * adv)), max(0.05, ba * math.exp(-ELO_SENSITIVITY * adv))


def blend_weight(fifa_home: float, fifa_away: float) -> float:
    return min(1.0, max(0.0, abs(fifa_home - fifa_away) / 400.0 / BLEND_SATURATION))


def calculate_poisson_probabilities(home_xg: float, away_xg: float, max_goals: int = 10) -> dict:
    ph = pd_ = pa = 0.0
    for i in range(max_goals):
        for j in range(max_goals):
            p = poisson.pmf(i, home_xg) * poisson.pmf(j, away_xg)
            if i > j: ph += p
            elif i == j: pd_ += p
            else: pa += p
    return {'home_win': ph * 100, 'draw': pd_ * 100, 'away_win': pa * 100}


def predict_match(home_team, away_team, neutral, model_home, model_away, df, fifa_ranking) -> dict:
    hs = get_team_stats(home_team, df); as_ = get_team_stats(away_team, df)
    hf = get_team_fifa(home_team, fifa_ranking); af = get_team_fifa(away_team, fifa_ranking)
    strength_diff = hs['strength'] - as_['strength']
    off_form_diff = hs['off'] - as_['off']
    def_form_diff = as_['def'] - hs['def']
    experience_ratio = math.log(hs['matches'] / as_['matches']) if hs['matches'] > 0 and as_['matches'] > 0 else 0.0
    fifa_diff = hf['points'] - af['points']
    inp = pd.DataFrame([{
        'home_off_form': hs['off'], 'away_off_form': as_['off'],
        'home_def_form': hs['def'], 'away_def_form': as_['def'],
        'home_strength': hs['strength'], 'away_strength': as_['strength'],
        'home_matches_played': hs['matches'], 'away_matches_played': as_['matches'],
        'strength_diff': strength_diff, 'off_form_diff': off_form_diff,
        'def_form_diff': def_form_diff, 'experience_ratio': experience_ratio,
        'home_team_tier': hs['tier'], 'away_team_tier': as_['tier'],
        'home_fifa_points': hf['points'], 'away_fifa_points': af['points'],
        'fifa_points_diff': fifa_diff, 'neutral': 1 if neutral else 0
    }])[FEATURE_COLUMNS]
    ml_h = max(0.0, float(model_home.predict(inp)[0]))
    ml_a = max(0.0, float(model_away.predict(inp)[0]))
    par_h, par_a = parametric_xg(hf['points'], af['points'], neutral)
    w = blend_weight(hf['points'], af['points'])
    xg_h = (1 - w) * ml_h + w * par_h
    xg_a = (1 - w) * ml_a + w * par_a
    probs = calculate_poisson_probabilities(xg_h, xg_a)
    return dict(hs=hs, as_=as_, hf=hf, af=af, ml_h=ml_h, ml_a=ml_a,
                par_h=par_h, par_a=par_a, w=w, xg_h=xg_h, xg_a=xg_a,
                probs=probs, fifa_diff=fifa_diff, strength_diff=strength_diff,
                off_form_diff=off_form_diff, def_form_diff=def_form_diff,
                experience_ratio=experience_ratio)


# ---------------------------------------------------------------------------
# HELPERS DE UI
# ---------------------------------------------------------------------------
def fmt_rank(info):
    return "Sin ranking (amateur)" if info['rank'] is None else f"#{info['rank']} · {info['points']:.0f} pts"


def favorite_badge(probs, home, away):
    if probs['home_win'] > 60: return f"🟢 {home}"
    if probs['away_win'] > 60: return f"🔴 {away}"
    return "🟡 Parejo"


def obtener_ultimos_partidos(team_name, df, n=5):
    tm = df[(df['home_team'] == team_name) | (df['away_team'] == team_name)].head(n)
    out = []
    for _, row in tm.iterrows():
        if row['home_team'] == team_name:
            goles, rival, local = row['home_score'], row['away_team'], True
        else:
            goles, rival, local = row['away_score'], row['home_team'], False
        if row['home_score'] == row['away_score']: res = "E"
        elif (local and row['home_score'] > row['away_score']) or (not local and row['away_score'] > row['home_score']): res = "G"
        else: res = "P"
        out.append({"Fecha": row["date"].strftime("%Y-%m-%d"), "Local": row["home_team"],
                    "Visitante": row["away_team"], "Goles": goles, "Rival": rival, "Resultado": res})
    return pd.DataFrame(out)


def go_to_prediction(home, away, neutral=False):
    """Navega a la página de detalle precargando el matchup (router interno).

    NO mutamos 'page' (key del radio) desde un callback: eso lanza
    WidgetAlreadyInstantiatedError. Escribimos un flag 'pending_page' que
    main() sincroniza ANTES de instanciar el widget en el próximo rerun.
    """
    st.session_state['sel_home'] = home
    st.session_state['sel_away'] = away
    st.session_state['sel_neutral'] = bool(neutral)
    st.session_state['pending_page'] = "Predicción"
    # token que fuerza la recarga de los selectboxes al llegar desde una card
    st.session_state['_nav_force'] = st.session_state.get('_nav_force', 0) + 1
    st.rerun()


# ---------------------------------------------------------------------------
# PÁGINA 1: ⚽ PARTIDOS (fixture hub estilo FotMob)
# ---------------------------------------------------------------------------
def page_partidos():
    model_home, model_away = load_models()
    df = load_data()
    fifa_ranking = load_fifa_ranking()
    teams_set = set(df["home_team"].unique())

    st.title("⚽ Jornada Sugerida")
    st.caption("Matchups destacados con predicción del motor Hybrid Ensemble. Tocá **Ver análisis** para el detalle.")
    st.divider()

    valid = [(h, a) for h, a in FIXTURES if h in teams_set and a in teams_set]
    for i in range(0, len(valid), 2):
        cols = st.columns(2)
        pair = valid[i:i + 2]
        for j in range(len(pair)):
            h, a = pair[j]
            with cols[j]:
                with st.container(border=True):
                    r = predict_match(h, a, False, model_home, model_away, df, fifa_ranking)
                    st.markdown(f"#### 🏠 {h}  vs  ✈️ {a}")
                    st.caption(f"🌍 {fmt_rank(r['hf'])}  ·  {fmt_rank(r['af'])}  |  Δ FIFA {r['fifa_diff']:+.0f}")
                    c1, c2, c3 = st.columns(3)
                    c1.markdown(f"**{r['probs']['home_win']:.0f}%**"); c1.progress(r['probs']['home_win'] / 100)
                    c2.markdown(f"**{r['probs']['draw']:.0f}%**"); c2.progress(r['probs']['draw'] / 100)
                    c3.markdown(f"**{r['probs']['away_win']:.0f}%**"); c3.progress(r['probs']['away_win'] / 100)
                    c1.caption("1 (local)"); c2.caption("X (empate)"); c3.caption("2 (visitante)")
                    st.markdown(f"⚽ xG: **{r['xg_h']:.2f} – {r['xg_a']:.2f}**  |  Favorito: {favorite_badge(r['probs'], h, a)}")
                    st.caption(f"⚖️ Blend: {r['w']*100:.0f}% param. / {(1-r['w'])*100:.0f}% ML")
                    if st.button("🔍 Ver análisis", key=f"fx_{i}_{j}", use_container_width=True):
                        go_to_prediction(h, a, False)


# ---------------------------------------------------------------------------
# PÁGINA 2: 🔮 PREDICCIÓN (detalle, con precarga desde Partidos/Equipos)
# ---------------------------------------------------------------------------
def page_prediccion():
    model_home, model_away = load_models()
    df = load_data()
    fifa_ranking = load_fifa_ranking()
    teams = sorted(df["home_team"].unique())

    st.title("🔮 Análisis de Partido")
    st.caption("Seleccioná los equipos (o llegá desde una card) y obtené el desglose completo del Hybrid Ensemble.")
    st.divider()

    # ¿Venimos de un click en una card? Entonces forzamos la precarga vía session_state.
    force = st.session_state.get('_nav_force', 0)
    seen = st.session_state.get('_nav_force_seen', None)
    applying = (force != seen)

    def_home = st.session_state.get('sel_home', "France")
    def_away = st.session_state.get('sel_away', "Arameans Suryoye")
    def_neut = bool(st.session_state.get('sel_neutral', False))

    if applying:
        st.session_state['home_sel'] = def_home if def_home in teams else teams[0]
        st.session_state['neut_sel'] = def_neut

    home_team = st.selectbox("🏠 Equipo Local", teams, key="home_sel")

    avail = [t for t in teams if t != home_team]
    if applying:
        aw = def_away if def_away in avail else avail[0]
        st.session_state['away_sel'] = aw
    # Blindaje: si el usuario cambió el local, el away guardado puede no existir ya.
    if 'away_sel' not in st.session_state or st.session_state['away_sel'] not in avail:
        st.session_state['away_sel'] = avail[0]
    away_team = st.selectbox("✈️ Equipo Visitante", avail, key="away_sel")

    neutral = st.checkbox("🏟️ ¿Cancha neutral?", key="neut_sel")

    if applying:
        st.session_state['_nav_force_seen'] = force

    r = predict_match(home_team, away_team, neutral, model_home, model_away, df, fifa_ranking)

    st.info(
        f"📊 **{home_team}** (Local) · {r['hs']['matches']} ptos | 🌍 {fmt_rank(r['hf'])}\n\n"
        f"- ⚽ Ataque: {r['hs']['off']:.2f} | 🛡️ Defensa: {r['hs']['def']:.2f} | 💪 Fuerza: {r['hs']['strength']*100:.1f}%\n\n"
        f"📊 **{away_team}** (Visitante) · {r['as_']['matches']} ptos | 🌍 {fmt_rank(r['af'])}\n\n"
        f"- ⚽ Ataque: {r['as_']['off']:.2f} | 🛡️ Defensa: {r['as_']['def']:.2f} | 💪 Fuerza: {r['as_']['strength']*100:.1f}%\n\n"
        f"📈 Δ FIFA: {r['fifa_diff']:+.0f} | ⚖️ Peso paramétrico: **{r['w']*100:.0f}%** (ML: {(1-r['w'])*100:.0f}%)"
    )

    if st.button("🔮 Predecir (Hybrid Ensemble)", type="primary", use_container_width=True):
        with st.spinner("Combinando modelo ML + calibración paramétrica..."):
            st.divider(); st.subheader(f"📈 Predicción: {home_team} vs {away_team}")
            st.markdown("### 🧬 Desglose del Hybrid Ensemble")
            b1, b2, b3 = st.columns(3)
            b1.metric("xG Modelo ML", f"{r['ml_h']:.2f} – {r['ml_a']:.2f}")
            b2.metric("xG Paramétrico (FIFA)", f"{r['par_h']:.2f} – {r['par_a']:.2f}")
            b3.metric("Blend aplicado", f"{r['w']*100:.0f}% param. / {(1-r['w'])*100:.0f}% ML")
            st.markdown("### ⚽ Goles Esperados (xG final)")
            c1, c2 = st.columns(2)
            c1.metric(f"{home_team}", f"{r['xg_h']:.2f}")
            c2.metric(f"{away_team}", f"{r['xg_a']:.2f}")
            st.info(f"🎯 **Marcador más probable:** {home_team} {round(r['xg_h'])} - {round(r['xg_a'])} {away_team}")
            st.divider(); st.markdown("### 📊 Probabilidades (Poisson)")
            m1, m2, m3 = st.columns(3)
            m1.metric(f"Victoria {home_team}", f"{r['probs']['home_win']:.1f}%")
            m2.metric("Empate", f"{r['probs']['draw']:.1f}%")
            m3.metric(f"Victoria {away_team}", f"{r['probs']['away_win']:.1f}%")
            if r['probs']['home_win'] > 75: st.success(f"🟢 **{home_team}** favorito abrumador ({r['probs']['home_win']:.1f}%).")
            elif r['probs']['home_win'] > 60: st.info(f"🔵 **{home_team}** claro favorito.")
            elif r['probs']['away_win'] > 75: st.error(f"🔴 **{away_team}** favorito abrumador ({r['probs']['away_win']:.1f}%).")
            elif r['probs']['away_win'] > 60: st.warning(f"🟠 **{away_team}** favorito.")
            else: st.warning("🟡 Partido parejo.")
            st.divider(); st.subheader("📋 Últimos 5 partidos")
            h1, h2 = st.columns(2)
            with h1:
                st.markdown(f"**{home_team}**")
                hh = obtener_ultimos_partidos(home_team, df)
                if not hh.empty: st.dataframe(hh, use_container_width=True, hide_index=True)
            with h2:
                st.markdown(f"**{away_team}**")
                aa = obtener_ultimos_partidos(away_team, df)
                if not aa.empty: st.dataframe(aa, use_container_width=True, hide_index=True)
            if "historial" not in st.session_state: st.session_state.historial = []
            st.session_state.historial.append({
                "Hora": datetime.now().strftime("%H:%M:%S"), "Local": home_team, "Visitante": away_team,
                "xG Local": f"{r['xg_h']:.2f}", "xG Visitante": f"{r['xg_a']:.2f}",
                "Prob. Local": f"{r['probs']['home_win']:.1f}%", "Blend param.": f"{r['w']*100:.0f}%",
            })

    if "historial" in st.session_state and st.session_state.historial:
        st.divider(); st.subheader("🕒 Historial de la sesión")
        st.dataframe(pd.DataFrame(st.session_state.historial), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# PÁGINA 3: 🏆 RANKING (tabla navegable estilo SofaScore)
# ---------------------------------------------------------------------------
def page_ranking():
    fifa_ranking = load_fifa_ranking()
    st.title("🏆 Ranking Mundial (FIFA oficial + ELO extendido)")
    st.caption("Top 20 con puntos FIFA oficiales (Wikipedia) + equipos 21-150 vía rating ELO convertido. Actualizable con `update_ranking.py`.")
    st.divider()

    rows = [{"Rank": v['rank'], "Equipo": k.title(), "Puntos": v['points']} for k, v in fifa_ranking.items()]
    rdf = pd.DataFrame(rows).sort_values("Rank")
    q = st.text_input("🔎 Buscar equipo", placeholder="ej: France, Argentina, Vietnam...")
    if q:
        rdf = rdf[rdf["Equipo"].str.contains(q, case=False, na=False)]
    st.dataframe(
        rdf, use_container_width=True, hide_index=True, height=560,
        column_config={
            "Rank": st.column_config.NumberColumn("Rank", format="#%d"),
            "Equipo": st.column_config.TextColumn("Equipo", width="medium"),
            "Puntos": st.column_config.ProgressColumn("Puntos FIFA/ELO", min_value=600, max_value=2050, format="%.0f"),
        },
    )


# ---------------------------------------------------------------------------
# PÁGINA 4: 👥 EQUIPOS (ficha + rivales sugeridos)
# ---------------------------------------------------------------------------
def page_equipos():
    model_home, model_away = load_models()
    df = load_data()
    fifa_ranking = load_fifa_ranking()
    teams = sorted(df["home_team"].unique())
    team_lookup = {_norm(t): t for t in teams}

    st.title("👥 Ficha de Equipo")
    team = st.selectbox("Elegí un equipo", teams, index=teams.index("Argentina") if "Argentina" in teams else 0)
    st.divider()

    s = get_team_stats(team, df)
    f = get_team_fifa(team, fifa_ranking)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Partidos (2010+)", s['matches'])
    c2.metric("Fuerza (K=200)", f"{s['strength']*100:.1f}%")
    c3.metric("Ataque rec.", f"{s['off']:.2f}")
    c4.metric("🌍 FIFA", fmt_rank(f))

    st.subheader("📋 Últimos 5 partidos")
    hist = obtener_ultimos_partidos(team, df)
    if not hist.empty: st.dataframe(hist, use_container_width=True, hide_index=True)

    st.subheader("🆚 Rivales sugeridos (como local)")
    ranked = sorted(fifa_ranking.items(), key=lambda kv: kv[1]['points'], reverse=True)
    rivals = []
    for k, _ in ranked:
        real = team_lookup.get(_norm(k))
        if real and real != team and real not in rivals:
            rivals.append(real)
        if len(rivals) >= 3:
            break
    # Fallback por si el ranking no casa suficientes equipos del dataset.
    if len(rivals) < 3:
        extra = sorted([t for t in teams if t != team],
                       key=lambda t: get_team_stats(t, df)['tier'], reverse=True)
        for t in extra:
            if t not in rivals:
                rivals.append(t)
            if len(rivals) >= 3:
                break

    for opp in rivals:
        with st.container(border=True):
            r = predict_match(team, opp, False, model_home, model_away, df, fifa_ranking)
            cc1, cc2, cc3, cc4 = st.columns([3, 1, 1, 1])
            cc1.markdown(f"**{team}** vs **{opp}**")
            cc2.markdown(f"1: {r['probs']['home_win']:.0f}%")
            cc3.markdown(f"X: {r['probs']['draw']:.0f}%")
            cc4.markdown(f"2: {r['probs']['away_win']:.0f}%")
            st.caption(f"xG {r['xg_h']:.2f}–{r['xg_a']:.2f} | {favorite_badge(r['probs'], team, opp)} | blend {r['w']*100:.0f}% param.")
            if st.button(f"🔍 Analizar {team} vs {opp}", key=f"eq_{opp}"):
                go_to_prediction(team, opp, False)


# ---------------------------------------------------------------------------
# PÁGINA 5: ℹ️ MODELO (arquitectura, stack, limitaciones, autor)
# ---------------------------------------------------------------------------
def page_modelo():
    st.title("ℹ️ Sobre el Modelo")
    st.markdown("""
**ScoutData** es un motor de predicción de fútbol que combina dos señales mediante un *blend* con gating por magnitud del matchup:

| Capa | Rol |
|------|-----|
| **1. ML (dual XGBoost)** | Predice xG local/visitante desde 18 features leak-free. Domina en partidos parejos. |
| **2. Paramétrica (Elo/FIFA)** | Mapea la diferencia de rating a goles esperados (función exponencial). Extrapola a extremos. |

El peso paramétrico crece con lo desbalanceado del cruce, y se expone por predicción (transparencia, no caja negra).

**Feature engineering:** forma ofensiva/defensiva (últimos 5), fuerza histórica con *shrinkage* bayesiano (K=200), team tier, experience ratio, diffs explícitos y puntos FIFA en vivo. Historiales actualizados *después* de cada partido → cero data leakage.

**Ranking auto-actualizable:** cascada Wikipedia (Top 20 oficial) + snapshot ELO (150 equipos, escala `0.85·ELO−300`) + fallback. Ejecutá `python src/data/update_ranking.py`.

**Limitaciones honestas:** el ranking es snapshot actual (no point-in-time histórico); equipos sin FIFA = 700 pts; sin alineaciones/lesiones; los tree models no extrapolan solos — de ahí la capa paramétrica (decisión de diseño, no parche).

**Stack:** Python · Pandas · XGBoost · Scikit-Learn · SciPy (Poisson) · Streamlit · BeautifulSoup.

⚽ Desarrollado por **Augusto Valentini** — Ingeniería en Informática (UADE).
""")


# ---------------------------------------------------------------------------
# ROUTER PRINCIPAL (single-file, sin st.navigation ni st.switch_page)
# ---------------------------------------------------------------------------
def main():
    if 'page' not in st.session_state:
        st.session_state['page'] = "Partidos"

    # Sincronizar navegación programática ANTES de crear el radio. Esto evita el
    # WidgetAlreadyInstantiatedError: el flag pending se aplica al state del widget
    # en lugar de mutarse desde un callback.
    if 'pending_page' in st.session_state:
        st.session_state['page'] = st.session_state.pop('pending_page')

    with st.sidebar:
        st.image("https://img.icons8.com/color/96/football2.png", width=80)
        st.title("ScoutData")
        st.caption("Football Analytics Hub")
        st.divider()
        st.radio(
            "Navegación",
            ["Partidos", "Predicción", "Ranking", "Equipos", "Modelo"],
            key="page",
            label_visibility="collapsed",
        )
        st.divider()
        st.markdown("⚽ *Desarrollado por Augusto Valentini*")

    pg = st.session_state['page']
    if pg == "Partidos":
        page_partidos()
    elif pg == "Predicción":
        page_prediccion()
    elif pg == "Ranking":
        page_ranking()
    elif pg == "Equipos":
        page_equipos()
    else:
        page_modelo()


if __name__ == "__main__":
    main()