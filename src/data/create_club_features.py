# src/data/create_club_features.py
"""
Feature engineering POINT-IN-TIME para el modelo de clubes (segunda cabeza del
Hybrid Ensemble). Lee data/raw/club_results.csv (del fetch) y produce
data/processed/club_matches_with_features.csv (que SI se commitea -> clone&run).

Diseno calibrado al diagnostico de datos (turno anterior):
  - Senal 1 (113/160 cold-start): ELO prior por debut = media de ELO de la misma
    liga en state, fallback 1500. NO 1500 global ciego.
  - Senal 2 (2a/2b falsos positivos): los 160 nombres son clubes DISTINTOS.
    NO hay mapeo de normalizacion (mergearia Man City con Man United). Uso tal cual.
  - Point-in-time estricto: features del partido i usan estado hasta i-1; el
    resultado de i actualiza state DESPUES (mismo principio anti-leakage de selecciones).
  - Ranking = ELO intra-liga (NO standings crudos: leakage + escala rota).
  - Features v1: ELO + forma + strength bayesiano + tier (SIN position/points de
    temporada -> roadmap v2). Margin ELO binario W/D/L (roadmap v2: goal-diff).

Limitaciones honestas documentadas:
  - ELO intra-liga only (football-data no tiene European comps -> no hay puente
    cross-league; un PL vs LaLiga no es comparable).
  - Liga AR de clubes fuera (fuente no la tiene).
  - Prior por debut usa media de liga en state, que incluye descendidos de
    temporadas previas (sesgo leve hacia abajo, ~10-20 pts, documentado).

Uso:
  python src/data/create_club_features.py
"""
import math
import logging
from collections import defaultdict, deque
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parents[2]
RAW_PATH = PROJECT_DIR / "data" / "raw" / "club_results.csv"
OUT_PATH = PROJECT_DIR / "data" / "processed" / "club_matches_with_features.csv"

FORM_WINDOW = 5
ELO_K = 20                 # factor K estandar de clubes (clubelo.com usa ~20-40)
ELO_BASE = 1500.0          # fallback prior si la liga no tiene nadie en state
STRENGTH_K = 200           # mismo shrinkage de selecciones

CLUB_FEATURE_COLUMNS = [
    'home_elo', 'away_elo', 'elo_diff',
    'home_off_form', 'away_off_form', 'off_form_diff',
    'home_def_form', 'away_def_form', 'def_form_diff',
    'home_strength', 'away_strength', 'strength_diff',
    'home_matches_played', 'away_matches_played', 'experience_ratio',
    'home_tier', 'away_tier',
    'neutral',
]


def _elo_expected(elo_a: float, elo_b: float) -> float:
    """Probabilidad esperada de que A venza a B (formula estandar ELO)."""
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))


def _prior_for_debut(league: str, state: dict) -> float:
    """Senal 1: prior = media de ELO de equipos de la MISMA liga ya en state.
    Fallback ELO_BASE si la liga no tiene nadie (primera temporada del dataset).
    Incluye descendidos de temporadas previas (sesgo leve documentado)."""
    elos = [s['elo'] for s in state.values() if s['league'] == league]
    if not elos:
        return ELO_BASE
    return sum(elos) / len(elos)


def _team_view(state: dict, team: str, is_home: bool, gf: int, ga: int):
    """Extrae la perspectiva de `team` en un partido (goles a favor/contra segun
    si fue local o visitante) y sus stats acumuladas point-in-time (estado ANTES
    de este partido, porque state aun no se actualizo con el resultado)."""
    s = state[team]
    form = list(s['form'])            # deque de (gf, ga) ultimos N
    off = sum(g for g, _ in form[-FORM_WINDOW:]) / max(1, len(form[-FORM_WINDOW:])) if form else 0.0
    dfn = sum(c for _, c in form[-FORM_WINDOW:]) / max(1, len(form[-FORM_WINDOW:])) if form else 0.0
    mp = s['mp']
    wins = s['w']
    # strength bayesiano con shrinkage (mismo que selecciones): un equipo nuevo
    # arranca con la tasa global de victorias de local, no con 0.
    strength = (wins + GLOBAL_WIN_RATE * STRENGTH_K) / (mp + STRENGTH_K)
    avg_goals = (s['gf_total'] / mp) if mp > 0 else 0.0
    tier = strength * 0.5 + avg_goals * 0.3 + math.log(mp + 1) * 0.02
    return {
        'elo': s['elo'], 'off': off, 'def': dfn, 'mp': mp,
        'strength': strength, 'tier': tier,
    }


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Loop secuencial en orden de fecha ASCENDENTE. Para cada partido: calcula
    features point-in-time (estado hasta D-1), las registra, y DESPUES actualiza
    state con el resultado (cero leakage)."""
    df = df.sort_values("Date").reset_index(drop=True)
    state = {}  # team -> {elo, mp, w, d, l, gf_total, ga_total, form(deque), league}
    rows = []

    for _, r in df.iterrows():
        h, a = r['HomeTeam'], r['AwayTeam']
        hg, ag = int(r['FTHG']), int(r['FTAG'])
        league = r['league']

        # --- cold-start: debut -> prior por liga (Senal 1) ---
        if h not in state:
            state[h] = {'elo': _prior_for_debut(league, state), 'mp': 0, 'w': 0, 'd': 0,
                        'l': 0, 'gf_total': 0, 'ga_total': 0, 'form': deque(maxlen=FORM_WINDOW),
                        'league': league}
        if a not in state:
            state[a] = {'elo': _prior_for_debut(league, state), 'mp': 0, 'w': 0, 'd': 0,
                        'l': 0, 'gf_total': 0, 'ga_total': 0, 'form': deque(maxlen=FORM_WINDOW),
                        'league': league}

        # --- features point-in-time (estado ANTES de este partido) ---
        hv = _team_view(state, h, True, hg, ag)
        av = _team_view(state, a, False, ag, hg)
        exp_ratio = math.log(hv['mp'] / av['mp']) if hv['mp'] > 0 and av['mp'] > 0 else 0.0
        rows.append({
            'Date': r['Date'], 'HomeTeam': h, 'AwayTeam': a,
            'FTHG': hg, 'FTAG': ag, 'league': league, 'season': r['season'],
            'home_elo': hv['elo'], 'away_elo': av['elo'],
            'elo_diff': hv['elo'] - av['elo'],
            'home_off_form': hv['off'], 'away_off_form': av['off'],
            'off_form_diff': hv['off'] - av['off'],
            'home_def_form': hv['def'], 'away_def_form': av['def'],
            'def_form_diff': av['def'] - hv['def'],
            'home_strength': hv['strength'], 'away_strength': av['strength'],
            'strength_diff': hv['strength'] - av['strength'],
            'home_matches_played': hv['mp'], 'away_matches_played': av['mp'],
            'experience_ratio': exp_ratio,
            'home_tier': hv['tier'], 'away_tier': av['tier'],
            'neutral': 0,   # ligas domesticas: sin cancha neutral (roadmap: copas)
        })

        # --- ACTUALIZACION point-in-time (DESPUES de features = cero leakage) ---
        # ELO binario W/D/L con K=20.
        exp_h = _elo_expected(state[h]['elo'], state[a]['elo'])
        actual_h = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
        state[h]['elo'] += ELO_K * (actual_h - exp_h)
        state[a]['elo'] += ELO_K * ((1.0 - actual_h) - (1.0 - exp_h))
        # forma + contadores
        state[h]['form'].append((hg, ag))
        state[a]['form'].append((ag, hg))
        for team, gf, ga, won, drew in ((h, hg, ag, hg > ag, hg == ag),
                                         (a, ag, hg, ag > hg, hg == ag)):
            s = state[team]
            s['mp'] += 1
            s['gf_total'] += gf
            s['ga_total'] += ga
            if won: s['w'] += 1
            elif drew: s['d'] += 1
            else: s['l'] += 1

    out = pd.DataFrame(rows)
    return out


def main():
    if not RAW_PATH.exists():
        logger.error("No existe %s. Corre primero: python src/data/fetch_club_results.py", RAW_PATH)
        raise SystemExit(1)

    df = pd.read_csv(RAW_PATH, parse_dates=['Date'])
    global GLOBAL_WIN_RATE
    GLOBAL_WIN_RATE = float((df['FTHG'] > df['FTAG']).mean())
    logger.info("Global home win rate (prior de strength): %.4f", GLOBAL_WIN_RATE)

    feats = build_features(df)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    feats.to_csv(OUT_PATH, index=False)

    # --- RESUMEN + DIAGNOSTICO DE CALIBRACION (para pieza 4: parametric_xg) ---
    ed = feats['elo_diff']
    print("\n=== club_matches_with_features.csv ===")
    print(f"filas: {len(feats)} (esperado = {len(df)} del raw)")
    print(f"equipos unicos (NO mergeados, Senal 2): {len(set(feats.HomeTeam)|set(feats.AwayTeam))}")
    print(f"NaN en features (debe ser 0): {feats[CLUB_FEATURE_COLUMNS].isna().sum().sum()}")
    print(f"\n--- DISTRIBUCION de elo_diff (INFO para calibrar parametric_xg en pieza 4) ---")
    print(f"min={ed.min():.0f}  p5={ed.quantile(.05):.0f}  p25={ed.quantile(.25):.0f}  "
          f"med={ed.median():.0f}  p75={ed.quantile(.75):.0f}  p95={ed.quantile(.95):.0f}  max={ed.max():.0f}")
    print(f"std={ed.std():.0f}  |diff|>100: {(ed.abs()>100).sum()}  |diff|>200: {(ed.abs()>200).sum()}")
    print(f"\n--- SANITY point-in-time: primeros 6 partidos de Arsenal (debe arrancar mp=0, off=0) ---")
    ars = feats[feats.HomeTeam == 'Arsenal'].head(6)
    print(ars[['Date', 'AwayTeam', 'home_matches_played', 'home_off_form', 'home_elo']].to_string(index=False))
    print(f"\n--- COLD-START: partidos donde algun equipo debut (mp=0) ---")
    debuts = feats[(feats.home_matches_played == 0) | (feats.away_matches_played == 0)]
    print(f"partidos de debut: {len(debuts)} ({len(debuts)/len(feats)*100:.1f}% del dataset)")
    print(f"Guardado en {OUT_PATH} (SI se commitea: data/processed/ no esta en .gitignore)")


if __name__ == "__main__":
    main()