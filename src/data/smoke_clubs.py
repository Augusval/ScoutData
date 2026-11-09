# src/data/smoke_clubs.py
"""
Smoke test del modelo de clubes (pieza 4).
Valida tres escenarios:
  1) Extremo intra-liga: Man City vs Norwich
  2) Parejo intra-liga: Arsenal vs Chelsea
  3) Cross-league: Man City vs Barcelona -> debe devolver None
"""
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

from src.engine.predictor_clubs import (
    load_club_models_pure,
    load_club_data_pure,
    predict_club_match,
)


def fmt_extreme(r):
    if r is None:
        return "None"
    return (
        f"xG {r['xg_h']:.2f}-{r['xg_a']:.2f} | "
        f"P(local) {r['probs']['home_win']:.1f}% | "
        f"blend {r['w']*100:.0f}% param | "
        f"elo_diff {r['elo_diff']:.0f}"
    )


def fmt_parejo(r):
    if r is None:
        return "None"
    return (
        f"xG {r['xg_h']:.2f}-{r['xg_a']:.2f} | "
        f"1/X/2 {r['probs']['home_win']:.0f}/"
        f"{r['probs']['draw']:.0f}/{r['probs']['away_win']:.0f} | "
        f"blend {r['w']*100:.0f}% param | "
        f"elo_diff {r['elo_diff']:.0f}"
    )


def main():
    mh, ma = load_club_models_pure()
    df = load_club_data_pure()

    extreme = predict_club_match("Man City", "Norwich", False, mh, ma, df)
    even = predict_club_match("Arsenal", "Chelsea", False, mh, ma, df)
    cross = predict_club_match("Man City", "Barcelona", False, mh, ma, df)

    print("EXTREMO Man City vs Norwich:", fmt_extreme(extreme))
    print("PAREJO Arsenal vs Chelsea:", fmt_parejo(even))
    print(
        "CROSS-LLEAGUE Man City vs Barcelona ->",
        "None (caption honesto, fuera de fuente)" if cross is None else "EXISTE?!",
    )


if __name__ == "__main__":
    main()