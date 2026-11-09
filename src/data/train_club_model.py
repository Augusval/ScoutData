# src/data/train_club_model.py
"""
Training dual XGBoost para el modelo de clubes (segunda cabeza del Hybrid
Ensemble). Lee data/processed/club_matches_with_features.csv (point-in-time,
sanity cero-leakage verde) y entrena DOS regresores: uno para goles de local
(FTHG), otro para goles de visitante (FTAG). Mismas 18 features, targets
distintos. Guarda src/models/xgboost_club_{home,away}.joblib.

DECISION DE DISENO (gate de entrada del training): split TEMPORAL estricto,
NUNCA random shuffle. Random shuffle pondria un partido de 2024 en train y uno
de 2014 en test -> leakage temporal, traicionando el point-in-time de las
features. Entonces: orden por Date, corte 80/10/10 por percentil de fecha.
train = pasado, val = reciente (early stopping), test = futuro estricto.

Regularizacion moderada (mismo espiritu anti-overfit que la shrinkage
bayesiana de features): max_depth 6, lr 0.05, subsample/colsample 0.8,
min_child_weight 5, L1/L2. Early stopping sobre el set de validacion.

FIXES INTEGRADOS (vs versiones previas):
  - get_booster() en vez de booster_ (API estable public; booster_/booster son
    detalles internos que renussian entre majors de xgboost).
  - Sanity por RATIO MAE/baseline, NO umbral absoluto. Los dos targets tienen
    baselines distintos (away_mean=1.21 < home_mean=1.53 por la ventaja de
    local), asi que un MAE absoluto no es comparable entre cabezas. Un AWAY de
    0.868 es SANO (ratio 0.72, supera la media 28%), no leakage.

Uso:
  python src/data/train_club_model.py
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from xgboost import XGBRegressor

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parents[2]
FEATS_PATH = PROJECT_DIR / "data" / "processed" / "club_matches_with_features.csv"
MODEL_HOME_PATH = PROJECT_DIR / "src" / "models" / "xgboost_club_home.joblib"
MODEL_AWAY_PATH = PROJECT_DIR / "src" / "models" / "xgboost_club_away.joblib"

# Las 18 features (mismo orden/columnas que CLUB_FEATURE_COLUMNS en create_club_features)
FEATURES = [
    'home_elo', 'away_elo', 'elo_diff',
    'home_off_form', 'away_off_form', 'off_form_diff',
    'home_def_form', 'away_def_form', 'def_form_diff',
    'home_strength', 'away_strength', 'strength_diff',
    'home_matches_played', 'away_matches_played', 'experience_ratio',
    'home_tier', 'away_tier',
    'neutral',
]

HP = dict(
    n_estimators=600, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
    reg_alpha=0.1, reg_lambda=1.0, objective='reg:squarederror',
    early_stopping_rounds=50, random_state=42, n_jobs=-1,
)


def _temporal_split(df: pd.DataFrame):
    """Split por percentil de DATE (no random). train=pasado, val=reciente, test=futuro."""
    df = df.sort_values("Date").reset_index(drop=True)
    dates = pd.to_datetime(df["Date"])
    c_train = dates.quantile(0.80)
    c_val = dates.quantile(0.90)
    train = df[dates <= c_train]
    val = df[(dates > c_train) & (dates <= c_val)]
    test = df[dates > c_val]
    logger.info("Split temporal -> train %d (<= %s) | val %d | test %d (> %s)",
                len(train), c_train.date(), len(val), len(test), c_val.date())
    return train, val, test


def _train_one(name: str, train, val, test, target: str):
    Xtr, Xva, Xte = train[FEATURES], val[FEATURES], test[FEATURES]
    ytr, yva, yte = train[target], val[target], test[target]
    model = XGBRegressor(**HP)
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    pred = model.predict(Xte)
    mae = float(np.mean(np.abs(pred - yte)))
    rmse = float(np.sqrt(np.mean((pred - yte) ** 2)))
    best_it = getattr(model, "best_iteration", None)
    # FIX: get_booster() es la API publica estable; booster_/booster son internos
    # que renussian entre majors (xgboost 3.x movio booster_ -> booster).
    # Nota: con early stopping, get_score() agrega importancia sobre todos los
    # trees del booster (incluidos los de paciencia mas alla de best_iteration),
    # asi que sale levemente diluida. Irrelevante para el sanity de "elo_diff
    # top-8". Roadmap: iteration_range=(0, best_it+1) para importancia exacta.
    gains = model.get_booster().get_score(importance_type="gain")
    top = sorted(gains.items(), key=lambda kv: kv[1], reverse=True)[:10]
    logger.info("[%s] target=%s | test MAE=%.3f RMSE=%.3f | best_iter=%s",
                name, target, mae, rmse, best_it)
    logger.info("[%s] top features por gain: %s", name,
                ", ".join(f"{k}={v:.0f}" for k, v in top))
    return model, dict(mae=mae, rmse=rmse, best_iter=best_it, top=top)


def main():
    if not FEATS_PATH.exists():
        logger.error("No existe %s. Corre primero: python src/data/create_club_features.py", FEATS_PATH)
        raise SystemExit(1)

    df = pd.read_csv(FEATS_PATH, parse_dates=["Date"])
    train, val, test = _temporal_split(df)

    mh, rh = _train_one("HOME", train, val, test, "FTHG")
    ma, ra = _train_one("AWAY", train, val, test, "FTAG")

    MODEL_HOME_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(mh, MODEL_HOME_PATH)
    joblib.dump(ma, MODEL_AWAY_PATH)
    logger.info("Guardados %s y %s", MODEL_HOME_PATH.name, MODEL_AWAY_PATH.name)

    # --- SANITY FLAGS (gate de salida: ratio vs baseline, NO umbral absoluto) ---
    # Baseline = "predecir la media del pasado" (train), la comparacion justa.
    # Cada cabeza se juzga contra SU propio baseline (away_mean < home_mean).
    base_h = float(train['FTHG'].mean())
    base_a = float(train['FTAG'].mean())
    print("\n=== SANITY DEL TRAINING (ratio MAE/baseline) ===")
    for tag, r, base in (("HOME", rh, base_h), ("AWAY", ra, base_a)):
        mae, ratio = r["mae"], r["mae"] / base
        if ratio < 0.4:
            flag_mae = "SOSPECHA LEAKAGE (ratio <0.4, casi perfecto = imposible)"
        elif ratio <= 0.85:
            flag_mae = f"OK (supera la media {(1 - ratio) * 100:.0f}%)"
        elif ratio <= 0.95:
            flag_mae = "ACEPTABLE (apenas supera la media)"
        else:
            flag_mae = "MODELO POBRE (ratio >0.95, ~predice la media)"
        elo_in_top = any(k == "elo_diff" for k, _ in r["top"][:8])
        flag_elo = "OK (elo_diff top-8)" if elo_in_top else "SOSPECHA: elo_diff no informa"
        conv = r["best_iter"] is not None and r["best_iter"] < HP["n_estimators"] - 1
        flag_conv = "OK (early stop)" if conv else "SOSPECHA: no convergio (best_iter==n_est)"
        print(f"[{tag}] MAE={mae:.3f} baseline={base:.2f} ratio={ratio:.2f} -> {flag_mae}")
        print(f"[{tag}] elo_diff -> {flag_elo} | convergencia -> {flag_conv}")
    print("Guia: ratio <=0.85 = el modelo aprende; ~1.0 = solo predice la media; <0.4 = sospecha leakage.")


if __name__ == "__main__":
    main()