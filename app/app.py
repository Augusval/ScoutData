"""
ScoutData AI V15 - Football Analytics Hub (UI + EN VIVO real + CLUBES fijos)
==========================================================================
Presentacion Streamlit. TODA la logica de prediccion vive en
src/engine/predictor.py (puro, sin streamlit). El en vivo vive en
src/data/fetch_live.py (stdlib, sin streamlit). La cabeza de clubes vive en
src/engine/predictor_clubs.py (puro, sin streamlit). Aca solo hay codigo de UI
(st.*), cache de presentacion y el router.

Paginas: ⚽ Jornada | ⭐ Destacados | 🏟️ Clubes | 🔮 Prediccion | 🏆 Ranking | 👥 Equipos | ℹ️ Modelo
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

# EN VIVO: src/data no es paquete, se accede por sys.path (mismo patron que
# predictor.py usa para traer fetch_fifa_ranking). fetch_live usa solo stdlib.
sys.path.insert(0, str(PROJECT_DIR / "src" / "data"))
from fetch_live import get_live_board, LIVE_STATUSES, FINISHED_STATUSES, PRE_STATUSES  # noqa: E402

# SEGUNDA CABEZA: modelo de clubes (intra-liga europeo).
from src.engine.predictor_clubs import (  # noqa: E402
    load_club_models_pure,
    load_club_data_pure,
    predict_club_match,
)

st.set_page_config(page_title="ScoutData — Football Analytics Hub", page_icon="⚽", layout="wide")


# ---------------------------------------------------------------------------
# CACHE UI (el motor y el fetch entregan versiones puras; streamlit las memoiza)
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


@st.cache_data(show_spinner=False, ttl=60)
def load_live_board():
    """Cachea el board 60s en memoria de Streamlit. La caché de disco de
    fetch_live ya tiene TTL 60; esta evita re-leer disco / re-agrupar en cada
    rerun de UI. 1 request cada 60s max (gracias al FIX E: board = live=all)."""
    return get_live_board()


@st.cache_resource(show_spinner=False)
def load_club_models_safe():
    """Carga modelos de clubes si existen; si no, devuelve None (degrada UI)."""
    try:
        return load_club_models_pure()
    except FileNotFoundError:
        return None


@st.cache_data(show_spinner=False, ttl=60)
def load_club_data_safe():
    """Carga features de clubes si existen; si no, devuelve None (degrada UI)."""
    try:
        return load_club_data_pure()
    except FileNotFoundError:
        return None


# ---------------------------------------------------------------------------
# NAVEGACION PROGRAMATICA (flag pending, NO se muta la key del radio)
# ---------------------------------------------------------------------------
def go_to_prediction(home, away, neutral=False):
    st.session_state['sel_home'] = home
    st.session_state['sel_away'] = away
    st.session_state['sel_neutral'] = bool(neutral)
    st.session_state['pending_page'] = "Predicción"
    st.session_state['_nav_force'] = st.session_state.get('_nav_force', 0) + 1
    st.rerun()


def go_to_clubs(home, away):
    st.session_state['club_home'] = home
    st.session_state['club_away'] = away
    st.session_state['pending_page'] = "Clubes"
    st.session_state['_club_nav_force'] = st.session_state.get('_club_nav_force', 0) + 1
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
# EN VIVO REAL (API-Football) — estructura v3: live + live_by_league (NO 'today')
# ---------------------------------------------------------------------------
def _status_chip(fx):
    s = fx.get("status", "TBD")
    if s in LIVE_STATUSES:
        el = fx.get("elapsed")
        return f"🟢 EN VIVO {el}'" if el else "🟢 EN VIVO"
    if s in FINISHED_STATUSES:
        return "✅ Final"
    if s in PRE_STATUSES:
        return f"🕐 {fx.get('kick_time') or '--:--'}"
    return f"⏸ {s}"


def render_live_card(fx, teams_set, model_home, model_away, df, fifa_ranking,
                     club_models, club_df, key):
    """Fila estilo FotMob: equipos + marcador/hora + chip de estado + analisis condicional.

    Orden de dominio:
      1) Si el motor de SELECCIONES conoce a ambos equipos -> boton nacional.
      2) Si no, intenta el motor de CLUBES (intra-liga europeo historico).
      3) Si ninguno puede, caption honesto. No inventamos numeros.
    """
    hs = fx.get("hs"); as_ = fx.get("as")
    score = f"{hs if hs is not None else '-'} : {as_ if as_ is not None else '-'}"
    with st.container(border=True):
        c1, c2, c3 = st.columns([3, 1.4, 3])
        c1.markdown(f"**{fx['home']}**")
        c2.markdown(
            f"<div style='text-align:center;font-size:1.25rem;font-weight:700'>{score}</div>",
            unsafe_allow_html=True,
        )
        c3.markdown(f"<div style='text-align:right'>**{fx['away']}**</div>", unsafe_allow_html=True)

        cc1, cc2 = st.columns([4, 1])
        cc1.caption(f"🏟️ {fx.get('venue') or '—'}  ·  {fx.get('kick_time') or ''}")
        cc2.markdown(_status_chip(fx))

        # 1) Motor de selecciones (dataset historico de nacionales)
        if fx["home"] in teams_set and fx["away"] in teams_set:
            if st.button("🔮 Predecir este duelo", key=f"lv_{key}", use_container_width=True):
                go_to_prediction(fx["home"], fx["away"], False)
            return

        # 2) Motor de clubes (intra-liga europeo, historic point-in-time)
        club_res = None
        if club_models is not None and club_df is not None:
            mh, ma = club_models
            try:
                club_res = predict_club_match(fx["home"], fx["away"], False, mh, ma, club_df)
            except Exception:
                club_res = None

        if club_res is not None:
            p = club_res["probs"]
            st.caption(
                f"🧠 Clubes: xG **{club_res['xg_h']:.2f}-{club_res['xg_a']:.2f}** | "
                f"1/X/2 **{p['home_win']:.0f}/{p['draw']:.0f}/{p['away_win']:.0f}** | "
                f"blend **{club_res['w']*100:.0f}%** param | "
                f"ELO diff **{club_res['elo_diff']:+.0f}**"
            )
            with st.expander("🔮 Ver análisis de clubes (último cruce histórico)", expanded=False):
                c1, c2, c3 = st.columns(3)
                c1.metric("xG local", f"{club_res['xg_h']:.2f}")
                c2.metric("xG visitante", f"{club_res['xg_a']:.2f}")
                c3.metric("Blend paramétrico", f"{club_res['w']*100:.0f}%")

                st.markdown(
                    f"📊 Probabilidades: **{p['home_win']:.1f}%** local / "
                    f"**{p['draw']:.1f}%** empate / **{p['away_win']:.1f}%** visitante"
                )
                st.caption(
                    f"🧠 ELO local {club_res['elo_h']:.0f} vs ELO visitante {club_res['elo_a']:.0f} "
                    f"(diff {club_res['elo_diff']:+.0f})"
                )
                st.caption(
                    f"📅 Último cruce en dataset: {club_res['date']} · {club_res['league']} · "
                    f"canonical: {club_res['home']} vs {club_res['away']}"
                )
                st.caption(
                    "⚠️ Limitación honesta: es un proxy histórico point-in-time del último cruce "
                    "disponible en football-data.co.uk, no un ELO en vivo 2026. Cross-league, "
                    "Champions/Europa League y Liga AR de clubes siguen fuera por fuente gratuita."
                )

            if st.button("🔗 Abrir en página Clubes", key=f"lvclub_{key}", use_container_width=True):
                go_to_clubs(club_res["home"], club_res["away"])
        else:
            # 3) Caption honesto: ni selecciones ni clubes pueden predecir este matchup.
            st.caption(
                "🔜 Fuera del modelo actual — selecciones no lo conocen y clubes "
                "no tiene histórico comparable en la fuente gratuita."
            )


def render_live_section(board, model_home, model_away, df, fifa_ranking, teams_set,
                        club_models, club_df):
    """Tira roja + secciones por liga, desde board['live_by_league'] (v3).

    Degrada elegante: sin key -> tarjeta de setup; quota/red -> caption; ninguna
    liga propia pitando ahora -> info con el count global (no se pinta firehose
    de las 77 comps del mundo, solo las que seguís, igual que FotMob).
    """
    meta = board.get("meta", {})
    if meta.get("degraded") == "no_key":
        st.info(
            "🔴 **Para ver la jornada EN VIVO real (marcador, minuto) configurá tu API Key gratis de "
            "[API-Football](https://dashboard.api-football.com):**\n\n"
            "```\n$env:SCOUTDATA_API_KEY = \"TU_KEY\"\n"
            "$env:SCOUTDATA_LEAGUE_IDS = \"128,1032,39,140,15\"\n```\n"
            "Descubrí ids con `python src/data/fetch_live.py --discover Argentina`. "
            "Mientras tanto, abajo tenés la **jornada histórica con predicciones** del motor de selecciones."
        )
        return

    groups = board.get("live_by_league", {})   # ← v3: live_by_league, NUNCA 'today'
    total = sum(len(rows) for rows in groups.values())
    if meta.get("degraded"):
        st.caption(f"⚠️ Fuente parcial ({meta['degraded']}) — mostrando lo disponible.")

    if total > 0:
        st.subheader(f"🔴 EN VIVO AHORA — {total} partidos en tus ligas")
        for lname, rows in groups.items():
            with st.expander(f"{lname}  ·  {len(rows)} en vivo", expanded=True):
                for i, fx in enumerate(rows):
                    render_live_card(
                        fx,
                        teams_set,
                        model_home,
                        model_away,
                        df,
                        fifa_ranking,
                        club_models,
                        club_df,
                        f"{lname}_{i}",
                    )
        st.divider()
    else:
        other = len(board.get("live", []))
        st.info(
            f"🔴 Ninguna de tus ligas de seguimiento tiene partidos pitando en este momento. "
            f"(Hay {other} partidos en vivo en otras competiciones del mundo.) "
            f"La grilla de 'programados no iniciados' (NS) del día requiere season 2026, behind paywall en free — "
            f"por eso acá solo vemos los que YA están pitando."
        )
        st.divider()


# ---------------------------------------------------------------------------
# PAGINA: ⚽ JORNADA (en vivo real arriba + jornada tematica de selecciones abajo)
# ---------------------------------------------------------------------------
def page_jornada():
    model_home, model_away = load_models()
    df = load_data()
    fifa_ranking = load_fifa_ranking()
    teams_set = set(df["home_team"].unique())

    club_models = load_club_models_safe()
    club_df = load_club_data_safe()

    st.title("⚽ Jornada")
    st.caption(
        "**Arriba:** jornada EN VIVO real vía API-Football (marcador + minuto de lo que está pitando ahora). "
        "**Abajo:** jornada histórica de selecciones con predicción del motor. "
        "También tenés la página **🏟️ Clubes** para analizar duelos europeos intra-liga sin depender del horario."
    )
    st.divider()

    board = load_live_board()
    render_live_section(
        board,
        model_home,
        model_away,
        df,
        fifa_ranking,
        teams_set,
        club_models,
        club_df,
    )

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
# PAGINA: ⭐ DESTACADOS (grilla 2xN de matchups curados)
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
# PAGINA: 🏟️ CLUBES (fija, independiente del en vivo)
# ---------------------------------------------------------------------------
def page_clubes():
    club_models = load_club_models_safe()
    club_df = load_club_data_safe()

    st.title("🏟️ Clubes — modelo intra-liga europeo")
    st.caption(
        "Segunda cabeza del Hybrid Ensemble. Usa ELO point-in-time calculado desde el histórico de "
        "football-data.co.uk (Premier League, La Liga, Serie A, Bundesliga, Ligue 1). "
        "No depende del en vivo: podés analizar cualquier cruce histórico disponible cuando quieras."
    )
    st.divider()

    if club_models is None or club_df is None:
        st.info(
            "🧱 Todavía no cargué los modelos de clubes. Para habilitar esta página, corré:\n\n"
            "```\npython src/data/fetch_club_results.py\n"
            "python src/data/create_club_features.py\n"
            "python src/data/train_club_model.py\n```\n"
            "Después volvé a abrir esta página."
        )
        return

    mh, ma = club_models
    clubs = sorted(set(club_df["HomeTeam"].unique()) | set(club_df["AwayTeam"].unique()))

    if not clubs:
        st.warning("No encontré clubes en el dataset procesado. Revisá `data/processed/club_matches_with_features.csv`.")
        return

    force = st.session_state.get('_club_nav_force', 0)
    seen = st.session_state.get('_club_nav_force_seen', None)
    applying = (force != seen)

    def_home = st.session_state.get('club_home', clubs[0])
    def_away = st.session_state.get('club_away', clubs[1] if len(clubs) > 1 else clubs[0])

    if applying:
        st.session_state['club_home_sel'] = def_home if def_home in clubs else clubs[0]

    home_team = st.selectbox("🏠 Equipo local (nombre canonical football-data)", clubs, key="club_home_sel")

    avail = [t for t in clubs if t != home_team]
    if not avail:
        st.warning("Necesito al menos dos clubes distintos para predecir.")
        return

    if applying:
        st.session_state['club_away_sel'] = def_away if def_away in avail else avail[0]

    if 'club_away_sel' not in st.session_state or st.session_state['club_away_sel'] not in avail:
        st.session_state['club_away_sel'] = avail[0]

    away_team = st.selectbox("✈️ Equipo visitante", avail, key="club_away_sel")

    if applying:
        st.session_state['_club_nav_force_seen'] = force

    st.caption(
        "💡 Los nombres acá son los canonical de football-data.co.uk (ej. `Man City`, `Paris SG`, `Ath Madrid`). "
        "La API y el en vivo aceptan alias tipo `Manchester City`, `Paris Saint-Germain`, `Atlético Madrid` "
        "y los normaliza automáticamente."
    )

    if st.button("🔮 Predecir (Hybrid Ensemble de clubes)", type="primary", use_container_width=True):
        r = predict_club_match(home_team, away_team, False, mh, ma, club_df)
        if r is None:
            st.session_state.pop('club_last', None)
            st.warning(
                "Este cruce no existe en el histórico intra-liga disponible. "
                "Puede ser cross-league, Champions/Europa League, Liga AR de clubes o un club nuevo. "
                "El modelo no inventa números: devuelve `None` y la UI muestra esta limitación honesta."
            )
        else:
            st.session_state['club_last'] = r

    # Mostrar resultado persistido solo si sigue vigente para el par seleccionado.
    if 'club_last' in st.session_state and st.session_state['club_last']:
        r = st.session_state['club_last']
        if r.get("home") != home_team or r.get("away") != away_team:
            st.session_state.pop('club_last', None)
        else:
            st.divider()
            st.subheader(f"📈 Predicción: {r['home']} vs {r['away']}")

            st.markdown("### 🧬 Desglose del Hybrid Ensemble de clubes")
            b1, b2, b3 = st.columns(3)
            b1.metric("xG Modelo ML", f"{r['ml_h']:.2f} – {r['ml_a']:.2f}")
            b2.metric("xG Paramétrico (ELO)", f"{r['par_h']:.2f} – {r['par_a']:.2f}")
            b3.metric("Blend aplicado", f"{r['w']*100:.0f}% param. / {(1-r['w'])*100:.0f}% ML")

            st.markdown("### ⚽ Goles Esperados (xG final)")
            c1, c2 = st.columns(2)
            c1.metric(f"{r['home']}", f"{r['xg_h']:.2f}")
            c2.metric(f"{r['away']}", f"{r['xg_a']:.2f}")

            st.info(f"🎯 **Marcador más probable:** {r['home']} {round(r['xg_h'])} - {round(r['xg_a'])} {r['away']}")

            st.divider()
            st.markdown("### 📊 Probabilidades (Poisson)")
            m1, m2, m3 = st.columns(3)
            m1.metric(f"Victoria {r['home']}", f"{r['probs']['home_win']:.1f}%")
            m2.metric("Empate", f"{r['probs']['draw']:.1f}%")
            m3.metric(f"Victoria {r['away']}", f"{r['probs']['away_win']:.1f}%")

            if r['probs']['home_win'] > 75:
                st.success(f"🟢 **{r['home']}** favorito abrumador ({r['probs']['home_win']:.1f}%).")
            elif r['probs']['home_win'] > 60:
                st.info(f"🔵 **{r['home']}** claro favorito.")
            elif r['probs']['away_win'] > 75:
                st.error(f"🔴 **{r['away']}** favorito abrumador ({r['probs']['away_win']:.1f}%).")
            elif r['probs']['away_win'] > 60:
                st.warning(f"🟠 **{r['away']}** favorito.")
            else:
                st.warning("🟡 Partido parejo.")

            st.divider()
            st.markdown("### 🧠 Contexto ELO y fuente")
            st.caption(
                f"ELO local: **{r['elo_h']:.0f}** | ELO visitante: **{r['elo_a']:.0f}** | "
                f"diff: **{r['elo_diff']:+.0f}**"
            )
            st.caption(
                f"📅 Proxy histórico: último cruce disponible en dataset: **{r['date']}** · **{r['league']}**"
            )
            st.caption(
                "⚠️ Limitación honesta: para un partido 2026 en vivo, esta página usa el último cruce histórico "
                "point-in-time del dataset como proxy. No es un ELO recalculado en vivo 2026. "
                "Cross-league, Champions/Europa League y Liga AR de clubes quedan fuera por limitación de fuente gratuita."
            )


# ---------------------------------------------------------------------------
# PAGINA: 🔮 PREDICCION (selecciones, detalle + desglose del blend + cara a cara)
# ---------------------------------------------------------------------------
def page_prediccion():
    model_home, model_away = load_models()
    df = load_data()
    fifa_ranking = load_fifa_ranking()
    teams = sorted(df["home_team"].unique())

    st.title("🔮 Análisis de Partido (Selecciones)")
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
# PAGINA: 🏆 RANKING
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
# PAGINA: 👥 EQUIPOS
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
# PAGINA: ℹ️ MODELO
# ---------------------------------------------------------------------------
def page_modelo():
    st.title("ℹ️ Sobre el Modelo")
    st.markdown("""
**ScoutData** combina dos señales mediante un *blend* con gating por magnitud del matchup:

| Capa | Rol |
|------|-----|
| **1. ML (dual XGBoost)** | Predice xG local/visitante desde features leak-free. Domina en partidos parejos. |
| **2. Paramétrica (FIFA/ELO o ELO clubes)** | Mapea la diferencia de rating a goles esperados (función exponencial). Extrapola a extremos. |

El peso paramétrico crece con lo desbalanceado del cruce y se expone por predicción (transparencia, no caja negra).

**Arquitectura desacoplada:** toda la lógica vive en `src/engine/predictor.py` (Python puro, **sin Streamlit**) y la segunda cabeza de clubes en `src/engine/predictor_clubs.py`. De modo que las consumen esta UI, una API REST (FastAPI) y tests unitarios. Acá (`app/app.py`) solo hay presentación y navegación.

**En vivo real:** la tira roja de Jornada consume `src/data/fetch_live.py` (API-Football, solo stdlib). El board v3 agrupa los partidos pitando **por liga en memoria** (1 request, no 6), porque el free tier solo accede a seasons 2022-2024 en el endpoint de grilla — `live=all` sí funciona.

**Clubes:** la página fija **🏟️ Clubes** y el endpoint `/v1/predict/club` usan `football-data.co.uk` (histórico largo, sin quota) y ELO intra-liga point-in-time. Predice matchups europeos intra-liga existentes en el histórico. Cross-league, Champions/Europa League y Liga AR de clubes quedan fuera por limitación de fuente gratuita. La UI muestra caption honesto cuando el modelo no puede predecir; no inventa números.

**Feature engineering:** forma ofensiva/defensiva (últimos 5), fuerza histórica con *shrinkage* bayesiano (K=200), team tier, experience ratio, diffs explícitos y puntos FIFA/ELO. Historiales actualizados *después* de cada partido → cero data leakage.

**Ranking auto-actualizable:** cascada Wikipedia (Top 20 oficial) + snapshot ELO (150 equipos, escala `0.85·ELO−300`) + fallback. Ejecutá `python src/data/update_ranking.py`.

**Limitaciones honestas:** ranking snapshot actual (no point-in-time); equipos sin FIFA = 700 pts; sin alineaciones/lesiones; los tree models no extrapolan solos — de ahí la capa paramétrica (decisión de diseño). La grilla de "programados no iniciados" (NS) del día requiere season 2026, behind paywall en free. El modelo de clubes usa proxy histórico del último cruce, no ELO en vivo 2026.

**Stack:** Python · Pandas · XGBoost · Scikit-Learn · SciPy (Poisson) · Streamlit · FastAPI · BeautifulSoup · API-Football.

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
                 ["Jornada", "Destacados", "Clubes", "Predicción", "Ranking", "Equipos", "Modelo"],
                 key="page", label_visibility="collapsed")
        st.divider()
        st.markdown("⚽ *Desarrollado por Augusto Valentini*")

    pg = st.session_state['page']
    if pg == "Jornada":
        page_jornada()
    elif pg == "Destacados":
        page_partidos()
    elif pg == "Clubes":
        page_clubes()
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