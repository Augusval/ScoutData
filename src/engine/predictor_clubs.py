# src/engine/predictor_clubs.py
"""
Segunda cabeza del Hybrid Ensemble: modelo de CLUBES (intra-liga).
================================================================
Predice partidos de clubes usando features point-in-time generadas por
create_club_features.py y modelos entrenados por train_club_model.py.

Filosofia:
  - Reusa el espiritu del ensemble de selecciones (parametric + gating + Poisson).
  - No inventa numeros para matchups que la fuente no puede comparar.
  - Cross-league / par inexistente -> None (caption honesto en UI/API).
  - ELO intra-liga only (football-data no tiene European comps).

LOOKUP: mask_fwd simple. En liga domestica cada par juega ida Y vuelta, asi que
la direccion que pedis (A vs B o B vs A) SIEMPRE existe como ida o vuelta en el
historico. No hace falta buscar en ambas direcciones ni swapear roles.

NORMALIZACION DE NOMBRES:
  API-Football y football-data.co.uk usan nombres distintos. Ej:
    Manchester City -> Man City
    Paris Saint-Germain -> Paris SG
    Atletico Madrid -> Ath Madrid
    AC Milan -> Milan
  Se usa un mapeo curado, no heuristica automatica, para evitar merges peligrosos
  como Man City vs Man United o Ath Bilbao vs Ath Madrid.

Uso:
  from src.engine.predictor_clubs import (
      load_club_models_pure, load_club_data_pure, predict_club_match,
      blend_weight_clubs, parametric_xg_clubs, canonical_club_name,
  )
"""
import sys
import math
import unicodedata
from pathlib import Path

import joblib
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

CLUB_MODEL_HOME_PATH = PROJECT_DIR / "src" / "models" / "xgboost_club_home.joblib"
CLUB_MODEL_AWAY_PATH = PROJECT_DIR / "src" / "models" / "xgboost_club_away.joblib"
CLUB_DATA_PATH = PROJECT_DIR / "data" / "processed" / "club_matches_with_features.csv"

CLUB_FEATURE_COLUMNS = [
    "home_elo", "away_elo", "elo_diff",
    "home_off_form", "away_off_form", "off_form_diff",
    "home_def_form", "away_def_form", "def_form_diff",
    "home_strength", "away_strength", "strength_diff",
    "home_matches_played", "away_matches_played", "experience_ratio",
    "home_tier", "away_tier",
    "neutral",
]

# Calibracion de clubes (derivada de la distribucion real de elo_diff:
# std ~131, extremo ~+/-460). selecciones usa FIFA /400 (extremo ~1259).
# Equivalencia de magnitud: 1259/400 ~= 460/150 -> divisor 150 para clubes.
ELO_DIVISOR = 150.0
BLEND_SATURATION = 3.0
BASE_HOME_GOALS = 1.45
BASE_AWAY_GOALS = 1.10
NEUTRAL_BASE = 1.28
ELO_SENSITIVITY = 0.25


def _norm_name(name: str) -> str:
    """Normaliza un nombre de equipo para comparar contra aliases.

    - quita acentos;
    - minusculas;
    - reemplaza & por 'and';
    - elimina puntuacion comun;
    - colapsa espacios.
    """
    if not name:
        return ""
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    s = s.lower()
    s = s.replace("&", " and ")
    for ch in "-.,'’()":
        s = s.replace(ch, " ")
    s = " ".join(s.split())
    return s


# Mapeo curado API-Football / nombres comunes -> canonical football-data.co.uk.
# No es exhaustivo, pero cubre los clubes mas frecuentes de las 5 ligas.
# Si un nombre no esta aqui, se usa tal cual. Si no existe en el CSV, -> None.
CLUB_ALIASES = {
    # ---------------- England ----------------
    "manchester city": "Man City",
    "man city": "Man City",
    "manchester united": "Man United",
    "man united": "Man United",
    "man utd": "Man United",
    "newcastle united": "Newcastle",
    "newcastle": "Newcastle",
    "nottingham forest": "Nott'm Forest",
    "nott m forest": "Nott'm Forest",
    "wolverhampton wanderers": "Wolves",
    "wolves": "Wolves",
    "brighton hove albion": "Brighton",
    "brighton": "Brighton",
    "tottenham hotspur": "Tottenham",
    "tottenham": "Tottenham",
    "spurs": "Tottenham",
    "west ham united": "West Ham",
    "west ham": "West Ham",
    "west bromwich albion": "West Brom",
    "west brom": "West Brom",
    "sheffield united": "Sheffield United",
    "leeds united": "Leeds",
    "leeds": "Leeds",
    "leicester city": "Leicester",
    "leicester": "Leicester",
    "southampton": "Southampton",
    "everton": "Everton",
    "fulham": "Fulham",
    "burnley": "Burnley",
    "brentford": "Brentford",
    "crystal palace": "Crystal Palace",
    "arsenal": "Arsenal",
    "chelsea": "Chelsea",
    "liverpool": "Liverpool",
    "watford": "Watford",
    "stoke city": "Stoke",
    "stoke": "Stoke",
    "sunderland": "Sunderland",
    "hull city": "Hull",
    "hull": "Hull",
    "norwich city": "Norwich",
    "norwich": "Norwich",
    "queens park rangers": "QPR",
    "qpr": "QPR",
    "cardiff city": "Cardiff",
    "cardiff": "Cardiff",
    "swansea city": "Swansea",
    "swansea": "Swansea",
    "afc bournemouth": "Bournemouth",
    "bournemouth": "Bournemouth",

    # ---------------- Spain ----------------
    "atletico madrid": "Ath Madrid",
    "atletico de madrid": "Ath Madrid",
    "atletico": "Ath Madrid",
    "athletic club": "Ath Bilbao",
    "athletic club de bilbao": "Ath Bilbao",
    "athletic bilbao": "Ath Bilbao",
    "ath bilbao": "Ath Bilbao",
    "real sociedad": "Sociedad",
    "real sociedad de futbol": "Sociedad",
    "sociedad": "Sociedad",
    "real betis": "Betis",
    "real betis balompie": "Betis",
    "betis": "Betis",
    "sevilla fc": "Sevilla",
    "sevilla": "Sevilla",
    "valencia cf": "Valencia",
    "valencia": "Valencia",
    "villarreal cf": "Villarreal",
    "villarreal": "Villarreal",
    "celta de vigo": "Celta",
    "celta": "Celta",
    "deportivo alaves": "Alaves",
    "alaves": "Alaves",
    "ud las palmas": "Las Palmas",
    "las palmas": "Las Palmas",
    "rcd mallorca": "Mallorca",
    "mallorca": "Mallorca",
    "ca osasuna": "Osasuna",
    "osasuna": "Osasuna",
    "rayo vallecano": "Vallecano",
    "rayo vallecano de madrid": "Vallecano",
    "vallecano": "Vallecano",
    "granada cf": "Granada",
    "granada": "Granada",
    "cadiz cf": "Cadiz",
    "cadiz": "Cadiz",
    "elche cf": "Elche",
    "elche": "Elche",
    "rcd espanyol": "Espanol",
    "espanyol": "Espanol",
    "espanol": "Espanol",
    "real valladolid": "Valladolid",
    "valladolid": "Valladolid",
    "sd eibar": "Eibar",
    "eibar": "Eibar",
    "cd leganes": "Leganes",
    "leganes": "Leganes",
    "deportivo la coruna": "La Coruna",
    "deportivo de la coruna": "La Coruna",
    "la coruna": "La Coruna",
    "sporting gijon": "Sp Gijon",
    "sporting de gijon": "Sp Gijon",
    "sp gijon": "Sp Gijon",
    "real madrid": "Real Madrid",
    "barcelona": "Barcelona",
    "fc barcelona": "Barcelona",
    "getafe": "Getafe",
    "girona": "Girona",
    "levante": "Levante",
    "huesca": "Huesca",
    "cordoba": "Cordoba",
    "malaga": "Malaga",

    # ---------------- Italy ----------------
    "ac milan": "Milan",
    "milan": "Milan",
    "internazionale": "Inter",
    "inter milan": "Inter",
    "fc internazionale milano": "Inter",
    "inter": "Inter",
    "juventus fc": "Juventus",
    "juventus": "Juventus",
    "as roma": "Roma",
    "roma": "Roma",
    "ss lazio": "Lazio",
    "societa sportiva lazio": "Lazio",
    "lazio": "Lazio",
    "ssc napoli": "Napoli",
    "societa sportiva calcio napoli": "Napoli",
    "napoli": "Napoli",
    "torino fc": "Torino",
    "torino football club": "Torino",
    "torino": "Torino",
    "atalanta bc": "Atalanta",
    "atalanta bergamasca calcio": "Atalanta",
    "atalanta": "Atalanta",
    "acf fiorentina": "Fiorentina",
    "fiorentina": "Fiorentina",
    "bologna fc": "Bologna",
    "bologna football club 1909": "Bologna",
    "bologna": "Bologna",
    "udinese calcio": "Udinese",
    "udinese": "Udinese",
    "genoa": "Genoa",
    "us sassuolo": "Sassuolo",
    "sassuolo": "Sassuolo",
    "hellas verona": "Verona",
    "verona": "Verona",
    "cagliari": "Cagliari",
    "empoli": "Empoli",
    "parma": "Parma",
    "venezia": "Venezia",
    "cremonese": "Cremonese",
    "lecce": "Lecce",
    "monza": "Monza",
    "salernitana": "Salernitana",
    "sampdoria": "Sampdoria",
    "spezia": "Spezia",
    "benevento": "Benevento",
    "crotone": "Crotone",
    "carpi": "Carpi",
    "cesena": "Cesena",
    "brescia": "Brescia",
    "pescara": "Pescara",
    "palermo": "Palermo",
    "frosinone": "Frosinone",

    # ---------------- Germany ----------------
    "bayern munchen": "Bayern Munich",
    "bayern munich": "Bayern Munich",
    "fc bayern munchen": "Bayern Munich",
    "bayern": "Bayern Munich",
    "borussia dortmund": "Dortmund",
    "dortmund": "Dortmund",
    "bayer 04 leverkusen": "Leverkusen",
    "leverkusen": "Leverkusen",
    "rb leipzig": "RB Leipzig",
    "rasenballsport leipzig": "RB Leipzig",
    "vfl wolfsburg": "Wolfsburg",
    "wolfsburg": "Wolfsburg",
    "eintracht frankfurt": "Ein Frankfurt",
    "frankfurt": "Ein Frankfurt",
    "sc freiburg": "Freiburg",
    "freiburg": "Freiburg",
    "vfb stuttgart": "Stuttgart",
    "stuttgart": "Stuttgart",
    "fc schalke 04": "Schalke 04",
    "schalke": "Schalke 04",
    "1 fsv mainz 05": "Mainz",
    "mainz 05": "Mainz",
    "mainz": "Mainz",
    "tsg hoffenheim": "Hoffenheim",
    "hoffenheim": "Hoffenheim",
    "borussia monchengladbach": "M'gladbach",
    "monchengladbach": "M'gladbach",
    "m gladbach": "M'gladbach",
    "fc augsburg": "Augsburg",
    "augsburg": "Augsburg",
    "werder bremen": "Werder Bremen",
    "werder": "Werder Bremen",
    "hertha bsc": "Hertha",
    "hertha berlin": "Hertha",
    "hertha": "Hertha",
    "1 fc koln": "FC Koln",
    "fc koln": "FC Koln",
    "koln": "FC Koln",
    "cologne": "FC Koln",
    "fortuna dusseldorf": "Fortuna Dusseldorf",
    "dusseldorf": "Fortuna Dusseldorf",
    "arminia bielefeld": "Bielefeld",
    "bielefeld": "Bielefeld",
    "greuther furth": "Greuther Furth",
    "vfl bochum": "Bochum",
    "bochum": "Bochum",
    "darmstadt 98": "Darmstadt",
    "darmstadt": "Darmstadt",
    "1 fc heidenheim": "Heidenheim",
    "heidenheim": "Heidenheim",
    "union berlin": "Union Berlin",
    "hannover 96": "Hannover",
    "hannover": "Hannover",
    "ingolstadt": "Ingolstadt",
    "paderborn": "Paderborn",
    "nurnberg": "Nurnberg",
    "1 fc nurnberg": "Nurnberg",
    "hamburger sv": "Hamburg",
    "hamburg": "Hamburg",

    # ---------------- France ----------------
    "paris saint germain": "Paris SG",
    "paris sg": "Paris SG",
    "psg": "Paris SG",
    "olympique de marseille": "Marseille",
    "olympique marseille": "Marseille",
    "marseille": "Marseille",
    "om": "Marseille",
    "olympique lyonnais": "Lyon",
    "lyon": "Lyon",
    "ol": "Lyon",
    "as monaco": "Monaco",
    "monaco": "Monaco",
    "lille osc": "Lille",
    "lille": "Lille",
    "stade rennais": "Rennes",
    "rennes": "Rennes",
    "rc lens": "Lens",
    "lens": "Lens",
    "ogc nice": "Nice",
    "nice": "Nice",
    "fc nantes": "Nantes",
    "nantes": "Nantes",
    "strasbourg alsace": "Strasbourg",
    "strasbourg": "Strasbourg",
    "toulouse fc": "Toulouse",
    "toulouse": "Toulouse",
    "montpellier hsc": "Montpellier",
    "montpellier": "Montpellier",
    "stade brestois 29": "Brest",
    "brest": "Brest",
    "aj auxerre": "Auxerre",
    "auxerre": "Auxerre",
    "angers sco": "Angers",
    "angers": "Angers",
    "stade de reims": "Reims",
    "reims": "Reims",
    "fc lorient": "Lorient",
    "lorient": "Lorient",
    "clermont foot 63": "Clermont",
    "clermont": "Clermont",
    "ac ajaccio": "Ajaccio",
    "ajaccio": "Ajaccio",
    "gfc ajaccio": "Ajaccio GFCO",
    "ajaccio gfco": "Ajaccio GFCO",
    "as saint etienne": "St Etienne",
    "saint etienne": "St Etienne",
    "st etienne": "St Etienne",
    "ea guingamp": "Guingamp",
    "guingamp": "Guingamp",
    "dijon fco": "Dijon",
    "dijon": "Dijon",
    "nimes olympique": "Nimes",
    "nimes": "Nimes",
    "girondins de bordeaux": "Bordeaux",
    "bordeaux": "Bordeaux",
    "metz": "Metz",
    "le havre ac": "Le Havre",
    "le havre": "Le Havre",
    "as nancy": "Nancy",
    "nancy": "Nancy",
    "evian thonon gaillard": "Evian Thonon Gaillard",
    "evian tg": "Evian Thonon Gaillard",
    "amiens sc": "Amiens",
    "amiens": "Amiens",
    "troyes": "Troyes",
    "bastia": "Bastia",
    "caen": "Caen",
}


def canonical_club_name(name: str) -> str:
    """Devuelve el nombre canonical de football-data si existe alias; si no, el original."""
    if not name:
        return name
    key = _norm_name(name)
    return CLUB_ALIASES.get(key, name)


def load_club_models_pure():
    if not CLUB_MODEL_HOME_PATH.exists():
        raise FileNotFoundError(
            f"Falta {CLUB_MODEL_HOME_PATH}. Corre primero: python src/data/train_club_model.py"
        )
    if not CLUB_MODEL_AWAY_PATH.exists():
        raise FileNotFoundError(
            f"Falta {CLUB_MODEL_AWAY_PATH}. Corre primero: python src/data/train_club_model.py"
        )
    return joblib.load(CLUB_MODEL_HOME_PATH), joblib.load(CLUB_MODEL_AWAY_PATH)


def load_club_data_pure() -> pd.DataFrame:
    if not CLUB_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Falta {CLUB_DATA_PATH}. Corre primero: python src/data/create_club_features.py"
        )
    df = pd.read_csv(CLUB_DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date", ascending=False).reset_index(drop=True)


def _load_poisson_fn():
    """Reusa calculate_poisson_probabilities del motor de selecciones si existe;
    si no, implementacion local equivalente (SciPy). Defensivo ante renombres."""
    try:
        from src.engine.predictor import calculate_poisson_probabilities as fn
        if callable(fn):
            return fn
    except Exception:
        pass

    from scipy.stats import poisson

    def _local_poisson_probs(xg_h: float, xg_a: float) -> dict:
        max_goals = 10
        ph = poisson.pmf(list(range(max_goals + 1)), xg_h)
        pa = poisson.pmf(list(range(max_goals + 1)), xg_a)
        home = draw = away = 0.0
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                p = float(ph[i] * pa[j])
                if i > j:
                    home += p
                elif i == j:
                    draw += p
                else:
                    away += p
        total = home + draw + away
        if total <= 0:
            return {"home_win": 0.0, "draw": 0.0, "away_win": 0.0}
        return {
            "home_win": home / total * 100.0,
            "draw": draw / total * 100.0,
            "away_win": away / total * 100.0,
        }
    return _local_poisson_probs


POISSON_FN = _load_poisson_fn()


def parametric_xg_clubs(elo_home: float, elo_away: float, neutral: bool) -> tuple:
    """Capa parametrica para clubes, calibrada a ELO intra-liga (divisor /150)."""
    bh, ba = (NEUTRAL_BASE, NEUTRAL_BASE) if neutral else (BASE_HOME_GOALS, BASE_AWAY_GOALS)
    adv = (elo_home - elo_away) / ELO_DIVISOR
    xg_h = max(0.05, bh * math.exp(ELO_SENSITIVITY * adv))
    xg_a = max(0.05, ba * math.exp(-ELO_SENSITIVITY * adv))
    return xg_h, xg_a


def blend_weight_clubs(elo_home: float, elo_away: float) -> float:
    """Gating por magnitud del matchup, calibrado a ELO de clubes (/150).
    diff=0 -> 0% parametrico (manda ML); extremo intra-liga -> cerca de 100%."""
    return min(1.0, max(0.0, abs(elo_home - elo_away) / ELO_DIVISOR / BLEND_SATURATION))


def predict_club_match(home_team: str, away_team: str, neutral: bool,
                       model_home, model_away, df: pd.DataFrame) -> dict | None:
    """
    Predice un matchup de clubes si existe en el dataset historico (mask_fwd),
    normalizando nombres API-Football -> football-data mediante aliases curados.

    Devuelve None si:
      - home == away despues de canonicalizar;
      - el par no existe en ESA direccion (cross-league, AR, club nuevo);
      - faltan modelos/datos (aunque eso se controla antes en la UI/API).

    La UI/API mostrara caption honesto. NO inventamos un numero.
    """
    home_can = canonical_club_name(home_team)
    away_can = canonical_club_name(away_team)

    if not home_can or not away_can or home_can == away_can:
        return None

    mask = (df["HomeTeam"] == home_can) & (df["AwayTeam"] == away_can)
    rows = df[mask]
    if rows.empty:
        return None

    row = rows.iloc[0]   # df ordenado Date desc -> ultimo encuentro en esa direccion

    feature_dict = {c: row[c] for c in CLUB_FEATURE_COLUMNS}
    inp = pd.DataFrame([feature_dict])[CLUB_FEATURE_COLUMNS]

    ml_h = max(0.0, float(model_home.predict(inp)[0]))
    ml_a = max(0.0, float(model_away.predict(inp)[0]))

    elo_h = float(row["home_elo"])
    elo_a = float(row["away_elo"])

    par_h, par_a = parametric_xg_clubs(elo_h, elo_a, neutral)
    w = blend_weight_clubs(elo_h, elo_a)
    xg_h = (1.0 - w) * ml_h + w * par_h
    xg_a = (1.0 - w) * ml_a + w * par_a
    probs = POISSON_FN(xg_h, xg_a)

    return {
        "home": home_can,
        "away": away_can,
        "home_input": home_team,
        "away_input": away_team,
        "neutral": bool(neutral),
        "ml_h": ml_h, "ml_a": ml_a,
        "par_h": par_h, "par_a": par_a,
        "w": w,
        "xg_h": xg_h, "xg_a": xg_a,
        "probs": probs,
        "elo_h": elo_h, "elo_a": elo_a, "elo_diff": elo_h - elo_a,
        "date": str(row["Date"].date()),
        "league": row["league"],
    }