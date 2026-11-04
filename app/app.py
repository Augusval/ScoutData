"""
ScoutData AI V12 - Football Analytics Hub (UI sobre el motor desacoplado)
==========================================================================
Presentación Streamlit. TODA la lógica de predicción vive en
src/engine/predictor.py (puro, sin streamlit) para que la consuman también
una API FastAPI y tests. Acá solo hay código de UI (st.*) y el router.

Páginas: ⚽ Jornada | ⭐ Destacados | 🔮 Predicción | 🏆 Ranking | 👥 Equipos | ℹ️ Modelo
Autor: Augusto Valentini
"""
import sys
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
from src.engine.predictor import (  # noqa: E402
    load_models_pure, load_data_pure, load_fifa_ranking_pure,
    get_team_stats, predict_match, get_head_to_head, get_confederation,
    get_flag, fmt_rank, favorite_badge, obtener_ultimos_partidos,
    suggest_rivals, get_team_fifa,
    FEATURE_COLUMNS, FIXTURES, JORNADAS, CONFED_LABEL, FORM_WINDOW,
)

st.set_page_config(page_title="ScoutData — Football Analytics Hub", page_icon="⚽", layout="wide")


# ---------------------------------------------------------------------------
# CACHE UI (el motor entrega versiones puras; streamlit las memoiza acá)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_models():
    return load_models_pure()


@st.cache_data(show_spinner=False, ttl=3600)
def load_data() -> pd.DataFrame:
    return load_data_pure()


@st.cache_data(show_spinner=False, ttl=86400)
def load_fifa_ranking() -> dict:
    return load_fifa_ranking_pure()


# ---------------------------------------------------------------------------
# NAVEGACIÓN PROGRAMÁTICA (flag pending, NO se muta la key del radio)
# ---------------------------------------------------------------------------
def go_to_prediction(home, away, neutral=False):
    st.session_state['sel_home'] = home
    st.session_state['sel_away'] = away
    st.session_state['sel_neutral'] = bool(neutral)
    st.session_state['pending_page'] = "Predicción"
    st.session_state['_nav_force'] = st.session_state.get('_nav_force', 0) + 1
    st.rerun()


def render_match_card(h, a, neutral, key_prefix, model_home, model_away, df, fifa_ranking):
    r = predict_match(h, a, neutral, model_home, model_away, df, fifa_ranking)
    with st.container(border=True):
        st.markdown(f"###### {get_flag(h)} {h}  ·  **vs**  ·  {a} {get_flag(a)}")
        st.caption(f"🌍 {fmt_rank(r['hf'])}  vs  {fmt_rank(r['af'])}  |  Δ FIFA {r['fifa_diff']:+.0f}")
        c1, c2, c3 = st.columns(3)
        c1.markdown(f"**{r['probs']['home_win']:.0f}%**"); c1.progress(r['probs']['home_win'] / 100)
        c2.markdown(f"**{r['probs']['draw']:.0f}%**"); c2.progress(r['probs']['draw'] / 100)
        c3.markdown(f"**{r['probs']['away_win']:.0f}%**"); c3.progress(r['probs']['away_win'] / 100)
        c1.caption("1 (local)"); c2.caption("X (empate)"); c3.caption("2 (visitante)")
        st.markdown(f"⚽ xG **{r['xg_h']:.2f} – {r['xg_a']:.2f}**  |  {favorite_badge(r['probs'], h, a)}  |  🧠 blend {r['w']*100:.0f}% param.")
        if st.button("🔍 Ver análisis", key=f"jcard_{key_prefix}", use_container_width=True):
            go_to_prediction(h, a, neutral)


# ---------------------------------------------------------------------------
# PÁGINA: ⚽ JORNADA (barra de jornadas + secciones por confederación)
# ---------------------------------------------------------------------------
def page_jornada():
    model_home, model_away = load_models()
    df = load_data()
    fifa_ranking = load_fifa_ranking()
    teams_set = set(df["home_team"].unique())

    st.title("⚽ Jornada")
    st.caption(
        "Elegí una jornada, explorá los duelos agrupados por confederación y abrí el análisis. "
        "**Nota honesta:** no hay marcadores en vivo ni clubes de liga — el motor es de selecciones "
        "y trabaja sobre histórico. Donde FotMob muestra “En vivo”, acá mostramos nuestra "
        "**predicción** (chip 🧠)."
    )
    st.divider()

    selected = st.radio("Jornada", list(JORNADAS.keys()), horizontal=True, label_visibility="collapsed")
    pairs = [(h, a) for h, a in JORNADAS[selected] if h in teams_set and a in teams_set]
    if not pairs:
        st.warning("Esta jornada no tiene equipos disponibles en el dataset.")
        return

    groups = {}
    for h, a in pairs:
        groups.setdefault(get_confederation(h), []).append((h, a))

    idx = 0
    for conf in sorted(groups.keys(), key=lambda c: (c == "No FIFA", c)):
        with st.expander(f"{CONFED_LABEL.get(conf, conf)}  ·  {len(groups[conf])} duelos", expanded=True):
            for h, a in groups[conf]:
                render_match_card(h, a, False, f"{selected}_{idx}", model_home, model_away, df, fifa_ranking)
                idx += 1


# ---------------------------------------------------------------------------
# PÁGINA: ⭐ DESTACADOS (grilla 2xN de matchups curados)
# ---------------------------------------------------------------------------
def page_partidos():
    model_home, model_away = load_models()
    df = load_data()
    fifa_ranking = load_fifa_ranking()
    teams_set = set(df["home_team"].unique())

    st.title("⭐ Partidos Destacados")
    st.caption("Matchups curados con predicción del motor Hybrid Ensemble.")
    st.divider()

    valid = [(h, a) for h, a in FIXTURES if h in teams_set and a in teams_set]
    for i in range(0, len(valid), 2):
        cols = st.columns(2)
        pair = valid[i:i + 2]
        for j in range(len(pair)):
            h, a = pair[j]
            with cols[j]:
                render_match_card(h, a, False, f"hl_{i}_{j}", model_home, model_away, df, fifa_ranking)


# ---------------------------------------------------------------------------
# PÁGINA: 🔮 PREDICCIÓN (detalle + desglose del blend + cara a cara)
# ---------------------------------------------------------------------------
def page_prediccion():
    model_home, model_away = load_models()
    df = load_data()
    fifa_ranking = load_fifa_ranking()
    teams = sorted(df["home_team"].unique())

    st.title("🔮 Análisis de Partido")
    st.caption("Seleccioná los equipos (o llegá desde una card) y obtené el desglose del Hybrid Ensemble.")
    st.divider()

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
        st.session_state['away_sel'] = def_away if def_away in avail else avail[0]
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

            st.divider(); st.subheader("🔁 Cara a cara (histórico directo)")
            h2h = get_head_to_head(home_team, away_team, df, n=6)
            if not h2h.empty:
                wins_h = (h2h["Ganador"] == home_team).sum()
                wins_a = (h2h["Ganador"] == away_team).sum()
                draws = (h2h["Ganador"] == "Empate").sum()
                st.markdown(f"📊 En los últimos {len(h2h)} cruces: **{home_team} {wins_h}** · **Empates {draws}** · **{away_team} {wins_a}**")
                st.dataframe(h2h, use_container_width=True, hide_index=True)
            else:
                st.info("Sin enfrentamientos históricos directos entre estos dos equipos en el dataset.")

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
# PÁGINA: 🏆 RANKING
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
# PÁGINA: 👥 EQUIPOS
# ---------------------------------------------------------------------------
def page_equipos():
    model_home, model_away = load_models()
    df = load_data()
    fifa_ranking = load_fifa_ranking()
    teams = sorted(df["home_team"].unique())

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
    for opp in suggest_rivals(team, df, fifa_ranking, teams, 3):
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
# PÁGINA: ℹ️ MODELO
# ---------------------------------------------------------------------------
def page_modelo():
    st.title("ℹ️ Sobre el Modelo")
    st.markdown("""
**ScoutData** combina dos señales mediante un *blend* con gating por magnitud del matchup:

| Capa | Rol |
|------|-----|
| **1. ML (dual XGBoost)** | Predice xG local/visitante desde 18 features leak-free. Domina en partidos parejos. |
| **2. Paramétrica (Elo/FIFA)** | Mapea la diferencia de rating a goles esperados (función exponencial). Extrapola a extremos. |

El peso paramétrico crece con lo desbalanceado del cruce y se expone por predicción (transparencia, no caja negra).

**Arquitectura desacoplada:** toda la lógica vive en `src/engine/predictor.py` (Python puro, **sin Streamlit**), de modo que la misma la consumen esta UI, una API REST (FastAPI) y tests unitarios. Acá (`app/app.py`) solo hay presentación y navegación.

**Feature engineering:** forma ofensiva/defensiva (últimos 5), fuerza histórica con *shrinkage* bayesiano (K=200), team tier, experience ratio, diffs explícitos y puntos FIFA en vivo. Historiales actualizados *después* de cada partido → cero data leakage.

**Ranking auto-actualizable:** cascada Wikipedia (Top 20 oficial) + snapshot ELO (150 equipos, escala `0.85·ELO−300`) + fallback. Ejecutá `python src/data/update_ranking.py`.

**Limitaciones honestas:** ranking snapshot actual (no point-in-time); equipos sin FIFA = 700 pts; sin alineaciones/lesiones; los tree models no extrapolan solos — de ahí la capa paramétrica (decisión de diseño).

**Stack:** Python · Pandas · XGBoost · Scikit-Learn · SciPy (Poisson) · Streamlit · BeautifulSoup.

⚽ Desarrollado por **Augusto Valentini** — Ingeniería en Informática (UADE).
""")


# ---------------------------------------------------------------------------
# ROUTER
# ---------------------------------------------------------------------------
def main():
    if 'page' not in st.session_state:
        st.session_state['page'] = "Jornada"
    if 'pending_page' in st.session_state:
        st.session_state['page'] = st.session_state.pop('pending_page')

    with st.sidebar:
        st.image("https://img.icons8.com/color/96/football2.png", width=80)
        st.title("ScoutData")
        st.caption("Football Analytics Hub")
        st.divider()
        st.radio("Navegación",
                 ["Jornada", "Destacados", "Predicción", "Ranking", "Equipos", "Modelo"],
                 key="page", label_visibility="collapsed")
        st.divider()
        st.markdown("⚽ *Desarrollado por Augusto Valentini*")

    pg = st.session_state['page']
    if pg == "Jornada":
        page_jornada()
    elif pg == "Destacados":
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