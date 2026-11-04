# ⚽ ScoutData — Hybrid Football Match Prediction Engine

An end-to-end Machine Learning system that predicts international football outcomes by combining **dual XGBoost regressors** with a **parametric Elo-calibration layer**, converting expected goals into realistic win/draw/loss probabilities via the **Poisson distribution**.

![Python](https://img.shields.io/badge/Python-3.11-blue) ![XGBoost](https://img.shields.io/badge/ML-XGBoost-orange) ![Streamlit](https://img.shields.io/badge/UI-Streamlit-red) ![License](https://img.shields.io/badge/License-MIT-green)

---

## 🎯 The Problem

Binary classifiers ("does the home team win?") fail on two fronts in football:
1. **Unrealistic extremes** — they cap around ~60–70% even in mismatches (e.g. France vs. a non-FIFA amateur side), because tree models cannot extrapolate beyond their training distribution.
2. **Scoreline blindness** — they ignore the structure of the game (goals, not just outcomes).

ScoutData solves both by predicting **expected goals (xG)** per team and deriving probabilities from a Poisson model — the same mathematical backbone used by professional betting syndicates and club analytics departments.

## ✨ Architecture: Hybrid Ensemble with Magnitude-Gated Blending

Every prediction fuses two independent signals:

| Layer | Source | Role |
|-------|--------|------|
| **1. ML Regressors** | Dual XGBoost (home xG, away xG) trained on 18 leak-free features | Dominates in evenly-matched games where training data is dense |
| **2. Parametric Elo** | Exponential mapping of FIFA rating difference → λ (goals) | Extrapolates correctly to extreme mismatches where trees cannot |

The blend weight is **gated by matchup magnitude**:

```
advantage = |fifa_home − fifa_away| / 400
w_param   = min(1.0, advantage / 3.0)
xG_final  = (1 − w_param) · xG_ML  +  w_param · xG_parametric
```

- **France vs. amateur** (Δ = 1259 pts) → `w_param = 100%` → **88.2%** home win ✅
- **Argentina vs. Spain** (Δ = 39 pts) → `w_param = 3%` → **41 / 28 / 30%** (realistic elite clash) ✅

The per-prediction blend is exposed in the UI, so the system is **self-explaining**, not a black box.

## 🧬 Feature Engineering (18 features, zero leakage)

- **Rolling form** (last 5 matches): offensive & defensive goals per game.
- **Historical strength** with **Bayesian shrinkage** `(wins + μ·K)/(matches + K)`, K=200 — prevents small-sample inflation for minor teams.
- **Team tier** & **experience ratio** (log of matches played).
- **Explicit diffs** (strength, attack, defense) so the model doesn't have to learn subtraction.
- **Live FIFA World Ranking** points (see below).

All team histories are updated **after** each match's features are computed, guaranteeing no future information leaks into training.

## 🌍 Self-Updating Ranking Pipeline

A cascading data pipeline keeps the FIFA signal fresh without hardcoding:

1. **Wikipedia** (BeautifulSoup) → official Top 20 FIFA points.
2. **Local ELO snapshot** (150 teams) → converted to FIFA scale via `0.85·ELO − 300`.
3. **Merge + fallback** → if any source fails, the last good JSON is reused.

Run manually: `python src/data/update_ranking.py`. Designed to be scheduled (Windows Task Scheduler / GitHub Actions cron) for fully autonomous weekly refresh.

## 📸 App Preview

The dashboard exposes, per prediction, the **full blend breakdown** (ML xG vs. parametric xG vs. applied weight), the final expected goals, the most-likely scoreline, and the Poisson win/draw/loss probabilities — plus each team's last 5 results. Run `streamlit run app/app.py` to see it live; screenshots coming in a follow-up commit.

## 🏗️ Project Structure

```
ScoutData/
├── app/app.py                      # Streamlit dashboard (inference + UI)
├── data/
│   ├── raw/                        # Kaggle dataset (gitignored)
│   ├── processed/                  # Cleaned + featured CSVs (gitignored)
│   └── external/fifa_ranking_2026.json   # Hybrid ranking (versioned)
├── src/
│   ├── data/
│   │   ├── clean_data.py           # Raw → processed
│   │   ├── create_features.py      # 18-feature engineering (leak-free)
│   │   ├── fetch_fifa_ranking.py   # Ranking loader
│   │   └── update_ranking.py       # Cascading self-updater
│   └── models/
│       ├── train_model.py          # Dual XGBoost regressor training
│       ├── xgboost_home_goals.joblib
│       └── xgboost_away_goals.joblib
└── README.md
```

## 🚀 Quick Start

```bash
python -m venv venv && venv\Scripts\activate   # Windows
pip install pandas numpy scikit-learn xgboost streamlit scipy beautifulsoup4 lxml

python src/data/update_ranking.py       # refresh FIFA ranking
python src/data/create_features.py      # build features
python src/models/train_model.py        # train dual regressors
streamlit run app/app.py                # launch dashboard
```

## 📊 Model Performance

| Metric | Value | Note |
|--------|-------|------|
| Home xG MAE | ~0.85 goals | Strong for football regression |
| Away xG MAE | ~0.80 goals | |
| Extreme match (FR vs amateur) | 88.2% home | Parametric layer engaged |
| Elite clash (AR vs ES) | 41 / 28 / 30 | ML layer engaged |

## ⚠️ Honest Limitations

- FIFA ranking is a **current snapshot** applied as a proxy for historical strength; a production system would use point-in-time ratings via the official API.
- Non-FIFA teams default to 700 points.
- No lineup, injury, or in-play data yet.
- Tree models cannot extrapolate alone — hence the parametric layer (a deliberate design choice, not a patch).

## 🗺️ Roadmap

- [ ] Point-in-time FIFA/ELO historical ratings
- [ ] Live fixtures ingestion (API-Football)
- [ ] GitHub Actions weekly auto-update of the ranking JSON
- [ ] Dockerized deployment

## 👤 Author

**Augusto Valentini** — Informatics Engineering student at UADE (Buenos Aires). Built as a portfolio project demonstrating production-grade Data Science: reproducible pipelines, leak-free feature engineering, hybrid modeling, and honest evaluation.