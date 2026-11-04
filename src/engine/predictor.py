# src/engine/predictor.py
"""
ScoutData ENGINE — motor de predicción puro, agnóstico de framework.
=====================================================================
Contiene TODA la lógica de dominio (features, Hybrid Ensemble, Poisson,
ranking, cara-a-cara, rivales) SIN ninguna dependencia de Streamlit.
Esto permite que la misma lógica la consuman la UI (app/app.py), una API
REST (FastAPI, fase siguiente) y tests unitarios, sin reescribir nada.

Autor: Augusto Valentini
"""
import sys
import math
from pathlib import Path

import pandas as pd
import joblib
from scipy.stats import poisson

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR / "src" / "data"))
from fetch_fifa_ranking import get_fifa_ranking, get_team_fifa  # noqa: E402

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

# Capa paramétrica (calibrada al fútbol internacional)
BASE_HOME_GOALS, BASE_AWAY_GOALS, NEUTRAL_BASE = 1.45, 1.10, 1.28
ELO_SENSITIVITY, BLEND_SATURATION = 0.25, 3.0

# Metadato de dominio (conocimiento del fútbol, NO dato inventado)
FIXTURES = [
    ("Spain", "France"), ("Argentina", "Brazil"), ("England", "Germany"),
    ("Portugal", "Netherlands"), ("Italy", "Belgium"), ("Morocco", "Japan"),
    ("France", "Arameans Suryoye"), ("Spain", "Afghanistan"), ("Argentina", "Arameans Suryoye"),
]
JORNADAS = {
    "Élite UEFA": [("Spain", "France"), ("England", "Germany"), ("Portugal", "Netherlands"),
                   ("Italy", "Belgium"), ("Spain", "Germany"), ("France", "Portugal")],
    "Clásicos CONMEBOL": [("Argentina", "Brazil"), ("Uruguay", "Argentina"), ("Chile", "Brazil"),
                          ("Colombia", "Argentina"), ("Brazil", "Uruguay")],
    "Intercontinentales": [("Morocco", "Japan"), ("Argentina", "France"), ("Brazil", "Germany"),
                           ("Spain", "Brazil"), ("France", "Argentina")],
    "Duelos extremos": [("France", "Arameans Suryoye"), ("Spain", "Afghanistan"),
                        ("Argentina", "Arameans Suryoye"), ("Brazil", "Afghanistan")],
}
CONFED = {
    "spain": "UEFA", "france": "UEFA", "england": "UEFA", "germany": "UEFA",
    "portugal": "UEFA", "netherlands": "UEFA", "italy": "UEFA", "belgium": "UEFA",
    "kosovo": "UEFA", "croatia": "UEFA", "switzerland": "UEFA", "denmark": "UEFA",
    "austria": "UEFA", "ukraine": "UEFA", "sweden": "UEFA", "poland": "UEFA",
    "serbia": "UEFA", "wales": "UEFA", "scotland": "UEFA", "greece": "UEFA",
    "hungary": "UEFA", "russia": "UEFA", "slovakia": "UEFA", "czechia": "UEFA",
    "romania": "UEFA", "norway": "UEFA", "iceland": "UEFA", "finland": "UEFA",
    "ireland": "UEFA", "albania": "UEFA", "georgia": "UEFA", "montenegro": "UEFA",
    "bosnia/herz": "UEFA", "bulgaria": "UEFA", "north macedonia": "UEFA",
    "slovenia": "UEFA", "moldova": "UEFA", "luxembourg": "UEFA", "latvia": "UEFA",
    "lithuania": "UEFA", "estonia": "UEFA", "cyprus": "UEFA", "malta": "UEFA",
    "andorra": "UEFA", "faroe islands": "UEFA", "armenia": "UEFA", "azerbaijan": "UEFA",
    "belarus": "UEFA", "turkiye": "UEFA", "israel": "UEFA",
    "argentina": "CONMEBOL", "brazil": "CONMEBOL", "chile": "CONMEBOL",
    "uruguay": "CONMEBOL", "colombia": "CONMEBOL", "ecuador": "CONMEBOL",
    "paraguay": "CONMEBOL", "peru": "CONMEBOL", "venezuela": "CONMEBOL", "bolivia": "CONMEBOL",
    "morocco": "CAF", "senegal": "CAF", "nigeria": "CAF", "egypt": "CAF", "algeria": "CAF",
    "tunisia": "CAF", "cote d'ivoire": "CAF", "cameroon": "CAF", "ghana": "CAF", "mali": "CAF",
    "burkina faso": "CAF", "cape verde": "CAF", "angola": "CAF", "zambia": "CAF", "uganda": "CAF",
    "dr congo": "CAF", "congo": "CAF", "gabon": "CAF", "south africa": "CAF", "guinea": "CAF",
    "benin": "CAF", "niger": "CAF", "togo": "CAF", "kenya": "CAF", "sudan": "CAF",
    "mauritania": "CAF", "comoros": "CAF", "madagascar": "CAF", "gambia": "CAF",
    "guinea-bissau": "CAF", "sierra leone": "CAF", "liberia": "CAF", "namibia": "CAF",
    "botswana": "CAF", "burundi": "CAF", "mozambique": "CAF", "zimbabwe": "CAF",
    "eswatini": "CAF", "lesotho": "CAF", "malawi": "CAF", "ethiopia": "CAF", "eritrea": "CAF",
    "djibouti": "CAF", "central african republic": "CAF", "chad": "CAF",
    "japan": "AFC", "ir iran": "AFC", "iran": "AFC", "saudi arabia": "AFC", "australia": "AFC",
    "uae": "AFC", "qatar": "AFC", "iraq": "AFC", "jordan": "AFC", "uzbekistan": "AFC",
    "china pr": "AFC", "china": "AFC", "korea republic": "AFC", "south korea": "AFC",
    "korea dpr": "AFC", "north korea": "AFC", "vietnam": "AFC", "thailand": "AFC",
    "indonesia": "AFC", "malaysia": "AFC", "kyrgyzstan": "AFC", "oman": "AFC", "bahrain": "AFC",
    "syria": "AFC", "palestine": "AFC", "india": "AFC", "tajikistan": "AFC",
    "turkmenistan": "AFC", "kuwait": "AFC", "lebanon": "AFC", "hong kong": "AFC",
    "philippines": "AFC", "myanmar": "AFC", "bangladesh": "AFC", "nepal": "AFC",
    "sri lanka": "AFC", "cambodia": "AFC", "taiwan": "AFC", "mongolia": "AFC",
    "afghanistan": "AFC", "maldives": "AFC", "bhutan": "AFC",
    "united states": "CONCACAF", "usa": "CONCACAF", "mexico": "CONCACAF", "canada": "CONCACAF",
    "costa rica": "CONCACAF", "panama": "CONCACAF", "jamaica": "CONCACAF", "honduras": "CONCACAF",
    "el salvador": "CONCACAF", "haiti": "CONCACAF", "trinidad/tobago": "CONCACAF",
    "curacao": "CONCACAF", "guatemala": "CONCACAF", "nicaragua": "CONCACAF", "cuba": "CONCACAF",
    "dominican republic": "CONCACAF", "suriname": "CONCACAF", "guyana": "CONCACAF",
    "bermuda": "CONCACAF", "barbados": "CONCACAF", "grenada": "CONCACAF",
    "antigua/barb": "CONCACAF", "saint lucia": "CONCACAF", "st vincent/g": "CONCACAF",
    "st kitts/n": "CONCACAF", "dominica": "CONCACAF", "aruba": "CONCACAF",
    "puerto rico": "CONCACAF", "martinique": "CONCACAF", "guadeloupe": "CONCACAF",
    "fr guiana": "CONCACAF", "cayman isl": "CONCACAF", "turks/caicos": "CONCACAF",
    "virgin isl": "CONCACAF", "anguilla": "CONCACAF", "montserrat": "CONCACAF",
    "new zealand": "OFC", "fiji": "OFC", "papua n g": "OFC", "tahiti": "OFC",
    "solomon isl": "OFC", "vanuatu": "OFC", "new caledonia": "OFC", "samoa": "OFC",
    "tonga": "OFC", "cook islands": "OFC", "niue": "OFC", "tuvalu": "OFC",
    "kiribati": "OFC", "marshall isl": "OFC", "palau": "OFC", "n marianas": "OFC",
    "american samoa": "OFC",
}
FLAGS = {
    "spain": "🇪", "france": "🇷", "england": "🏴󠁢󠁮󠁿", "germany": "🇩",
    "portugal": "🇵🇹", "netherlands": "🇳", "italy": "🇹", "belgium": "🇧🇪",
    "argentina": "🇦", "brazil": "🇷", "morocco": "🇲", "japan": "🇯🇵",
    "united states": "🇺", "mexico": "🇲🇽", "chile": "🇨", "uruguay": "🇾",
    "colombia": "🇨🇴", "senegal": "🇸", "korea republic": "🇰", "australia": "🇦🇺",
    "afghanistan": "🇦🇫", "iran": "🇮", "ir iran": "🇷", "saudi arabia": "🇸🇦",
    "canada": "🇨🇦", "costa rica": "🇨🇷", "nigeria": "🇳🇬", "egypt": "🇪🇬",
    "croatia": "🇭🇷", "switzerland": "🇨", "denmark": "🇰", "norway": "🇳🇴",
    "poland": "🇵🇱", "serbia": "🇷🇸", "ukraine": "🇺", "sweden": "🇸🇪",
    "turkiye": "🇹🇷", "tunisia": "🇹🇳", "cameroon": "🇨🇲", "ghana": "🇬",
    "peru": "🇵🇪", "paraguay": "🇵🇾", "ecuador": "🇪", "venezuela": "🇻🇪",
    "bolivia": "🇧🇴", "jamaica": "🇯🇲", "panama": "🇵🇦", "honduras": "🇭🇳",
    "new zealand": "🇳🇿", "qatar": "🇶", "iraq": "🇶", "oman": "🇴🇲",
    "bahrain": "🇧🇭", "uae": "🇦🇪", "china pr": "🇨", "korea dpr": "🇰🇵",
    "vietnam": "🇻🇳", "thailand": "🇹", "indonesia": "🇮🇩", "malaysia": "🇲🇾",
    "india": "🇮🇳", "uzbekistan": "🇺🇿", "jordan": "🇯🇴", "syria": "🇸",
    "palestine": "🇵", "algeria": "🇩🇿", "ivory coast": "🇨🇮", "cote d'ivoire": "🇨",
    "mali": "🇲🇱", "burkina faso": "🇧🇫", "cape verde": "🇨🇻", "angola": "🇦🇴",
    "zambia": "🇿🇲", "uganda": "🇺", "dr congo": "🇨", "congo": "🇬",
    "gabon": "🇬🇦", "south africa": "🇿", "guinea": "🇳", "wales": "🏴󠁢󠁬󠁳",
    "scotland": "🏴󠁧󠁢󠁳󠁴", "ireland": "🇪", "greece": "🇬", "hungary": "🇺",
    "russia": "🇷🇺", "slovakia": "🇸🇰", "czechia": "🇨🇿", "romania": "🇷🇴",
    "austria": "🇦🇹", "iceland": "🇮", "finland": "🇮", "albania": "🇦🇱",
    "georgia": "🇬", "montenegro": "🇪", "bulgaria": "🇧🇬", "kosovo": "🇽",
    "north macedonia": "🇲🇰", "slovenia": "🇸", "belarus": "🇾", "armenia": "🇦🇲",
    "azerbaijan": "🇦🇿", "israel": "🇮", "cyprus": "🇾", "malta": "🇲🇹",
    "luxembourg": "🇱🇺", "latvia": "🇱🇻", "lithuania": "🇱🇹", "estonia": "🇪",
    "moldova": "🇲", "faroe islands": "🇫🇴", "andorra": "🇦🇩",
    "trinidad/tobago": "🇹🇹", "curacao": "🇨🇼", "el salvador": "🇸🇻",
    "guatemala": "🇬🇹", "nicaragua": "🇳🇮", "cuba": "🇨", "haiti": "🇹",
    "dominican republic": "🇩🇴", "suriname": "🇸", "guyana": "🇬🇾",
    "bermuda": "🇧", "barbados": "🇧", "grenada": "🇩", "puerto rico": "🇵🇷",
    "fiji": "🇫🇯", "tahiti": "🇵", "papua n g": "🇵", "solomon isl": "🇸🇧",
    "vanuatu": "🇻", "new caledonia": "🇳🇨", "bosnia/herz": "🇧",
    "arameans suryoye": "🏳️",
}
CONFED_LABEL = {
    "UEFA": "🇪🇺 UEFA (Europa)", "CONMEBOL": "🛰️ CONMEBOL (Sudamérica)",
    "CAF": "🌍 CAF (África)", "AFC": "🌏 AFC (Asia / Oceanía)",
    "CONCACAF": "🌎 CONCACAF (Norteamérica)", "OFC": "🏝️ OFC (Oceanía)",
    "No FIFA": "🏳️ No FIFA / Diáspora",
}


# ---------------------------------------------------------------------------
# NORMALIZACIÓN
# ---------------------------------------------------------------------------
def _norm(s: str) -> str:
    import unicodedata
    s = unicodedata.normalize('NFKD', str(s))
    s = ''.join(c for c in s if not unicodedata.combining(c))
    return s.lower().replace('.', '').replace('-', ' ').strip()


# ---------------------------------------------------------------------------
# LOADERS PUROS (sin @st.cache — la UI los cachea por su cuenta)
# ---------------------------------------------------------------------------
def load_models_pure():
    return joblib.load(MODEL_HOME_PATH), joblib.load(MODEL_AWAY_PATH)


def load_data_pure() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"].dt.year >= 2010].copy()
    return df.sort_values("date", ascending=False).reset_index(drop=True)


def load_fifa_ranking_pure() -> dict:
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
# MOTOR: HYBRID ENSEMBLE
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
# DOMINIO + HELPERS PUROS
# ---------------------------------------------------------------------------
def get_confederation(team: str) -> str:
    return CONFED.get(_norm(team), "No FIFA")


def get_flag(team: str) -> str:
    return FLAGS.get(_norm(team), "🏳️")


def get_head_to_head(home: str, away: str, df: pd.DataFrame, n: int = 6) -> pd.DataFrame:
    mask = ((df['home_team'] == home) & (df['away_team'] == away)) | \
           ((df['home_team'] == away) & (df['away_team'] == home))
    h2h = df[mask].sort_values('date', ascending=False).head(n)
    out = []
    for _, row in h2h.iterrows():
        if row['home_score'] == row['away_score']:
            ganador = "Empate"
        elif row['home_score'] > row['away_score']:
            ganador = row['home_team']
        else:
            ganador = row['away_team']
        out.append({
            "Fecha": row['date'].strftime("%Y-%m-%d"),
            "Local": row['home_team'], "Marcador": f"{row['home_score']}-{row['away_score']}",
            "Visitante": row['away_team'], "Ganador": ganador,
        })
    return pd.DataFrame(out)


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


def suggest_rivals(team, df, fifa_ranking, teams, n=3):
    team_lookup = {_norm(t): t for t in teams}
    ranked = sorted(fifa_ranking.items(), key=lambda kv: kv[1]['points'], reverse=True)
    rivals = []
    for k, _ in ranked:
        real = team_lookup.get(_norm(k))
        if real and real != team and real not in rivals:
            rivals.append(real)
        if len(rivals) >= n: break
    if len(rivals) < n:
        extra = sorted([t for t in teams if t != team],
                       key=lambda t: get_team_stats(t, df)['tier'], reverse=True)
        for t in extra:
            if t not in rivals: rivals.append(t)
            if len(rivals) >= n: break
    return rivals


def fmt_rank(info):
    return "Sin ranking (amateur)" if info['rank'] is None else f"#{info['rank']} · {info['points']:.0f} pts"


def favorite_badge(probs, home, away):
    if probs['home_win'] > 60: return f"🟢 {home}"
    if probs['away_win'] > 60: return f"🔴 {away}"
    return "🟡 Parejo"