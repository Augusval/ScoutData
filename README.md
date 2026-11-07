# ⚽ ScoutData — Hybrid Football Prediction Engine (ML + REST API)

End-to-end system that predicts international football outcomes by fusing **dual XGBoost regressors** with a **parametric Elo-calibration layer**, turning expected goals into realistic 1X2 probabilities via the **Poisson distribution**. Shipped as a **multi-page Streamlit dashboard** *and* a **contract-first FastAPI service** that share a single, framework-agnostic engine.

![Python](https://img.shields.io/badge/Python-3.15-blue) ![XGBoost](https://img.shields.io/badge/ML-XGBoost-orange) ![FastAPI](https://img.shields.io/badge/API-FastAPI-009688) ![Streamlit](https://img.shields.io/badge/UI-Streamlit-red) ![pytest](https://img.shields.io/badge/tests-16%20passed-brightgreen) ![License](https://img.shields.io/badge/License-MIT-green)

---

## 🎯 The problem

Binary classifiers ("does the home team win?") fail twice in football:
1. **Unrealistic extremes** — they cap near ~60–70% even in mismatches (e.g. France vs. a non-FIFA side), because tree models cannot extrapolate beyond their training distribution.
2. **Scoreline blindness** — they ignore the structure of the game (goals, not just outcomes).

ScoutData solves both by predicting **expected goals (xG)** per team and deriving probabilities from a Poisson model — the mathematical backbone used by professional betting syndicates and club analytics departments.

## 🧬 Architecture: a hybrid ensemble with magnitude-gated blending

Every prediction fuses two independent signals:

| Layer | Source | Role |
|-------|--------|------|
| **1. ML regressors** | Dual XGBoost (home xG, away xG) on 18 leak-free features | Dominates in evenly-matched games where training data is dense |
| **2. Parametric Elo** | Exponential map of FIFA rating difference → λ (goals) | Extrapolates correctly to extreme mismatches where trees cannot |

The blend weight is **gated by matchup magnitude**:

```
advantage = |fifa_home − fifa_away| / 400
w_param   = min(1.0, advantage / 3.0)
xG_final  = (1 − w_param) · xG_ML  +  w_param · xG_parametric
```

Validated on two extremes, in the same build:

| Matchup | Δ FIFA | w_param | 1 / X / 2 | Reading |
|---------|--------|---------|-----------|---------|
| France vs. amateur | +1259 | **100%** | 88.2 / 8.4 / 3.2 | ML has no data → formula leads |
| Argentina vs. Spain (#2 vs #1) | −39 | **3%** | 41 / 28 / 30 | ML has thousands of examples → it leads |

The per-prediction blend is exposed in both the UI and the API, so the system is **self-explaining**, not a black box.

## 🔌 One engine, two fronts (the decoupling payoff)

All domain logic lives in [`src/engine/predictor.py`](src/engine/predictor.py) — **pure Python, zero Streamlit imports**. That single module is consumed by:

- the **Streamlit dashboard** (`app/app.py`) — presentation + routing only;
- the **FastAPI service** (`api_fastapi.py`) — HTTP contract only;
- the **pytest suite** (`tests/`) — runs against the engine *and* the API.

A dedicated test (`test_convergence_motor_equals_api`) proves the engine and the HTTP endpoint return **identical numbers** (88.2 / 3.18 / 100%) despite loading the models separately — so UI and backend cannot silently drift. This is the concrete reward of having decoupled the engine first.

## 🌐 REST API (contract-first)

`POST /v1/predict` validates input (Pydantic) **and** output (`response_model`), injects the engine via `Depends` + `lru_cache` (no mutable global), and **filters undeclared fields** from the response. Interactive OpenAPI docs at `/docs`.

```bash
curl -X POST http://127.0.0.1:8000/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"home":"France","away":"Arameans Suryoye"}'
```

Returns a typed JSON with `xg.{ml,parametric,final}`, `blend.parametric_pct`, `probabilities_pct` and `most_likely_score`. Unknown teams → `404`; same team / missing field → `422`.

## 🧪 Tests (16, green, zero-noise)

`pytest -v` runs the engine properties (extreme vs. even gating, ISO2 flags, head-to-head shape) and the API contract (404/422, singleton caching, response filtering, convergence). `pytest.ini` sets `max_warnings = 0` as a **ratchet**: any new warning fails the run (exit 6), while the one known third-party deprecation (Starlette) is filtered by an anchored regex — so filtered warnings don't count toward the maximum. The suite is a drift canary for both your code *and* the dependencies you silence.

## 🧱 Feature engineering (18 features, zero leakage)

- Rolling form (last 5): offensive & defensive goals/game.
- Historical strength with **Bayesian shrinkage** `(wins + μ·K)/(matches + K)`, K=200 — prevents small-sample inflation for minor teams.
- Team tier & experience ratio (log of matches played).
- Explicit diffs (strength, attack, defense) so the model doesn't relearn subtraction.
- Live FIFA World Ranking points (see below).

Team histories are updated **after** each match's features are computed → no future information leaks into training.

## 🌍 Self-updating ranking pipeline

A cascading pipeline keeps the FIFA signal fresh without hardcoding:

1. **Wikipedia** (BeautifulSoup) → official Top 20 FIFA points.
2. **Local ELO snapshot** (150 teams) → converted to FIFA scale via `0.85·ELO − 300`.
3. **Merge + fallback** → if any source fails, the last good JSON is reused.

Run `python src/data/update_ranking.py`. Designed to be scheduled (Task Scheduler / GitHub Actions cron) for autonomous weekly refresh.

## 📱 Dashboard (multi-page, FotMob-style)

`streamlit run app/app.py` opens a navigable hub — **Jornada** (themed matchdays grouped by confederation, 1X2 bars inline), **Predicción** (full blend breakdown + Poisson + last-5 + head-to-head), **Ranking** (searchable table with progress bars), **Equipos** (team card + suggested rivals), **Modelo** (architecture & honest limits). Programmatic navigation uses a `pending_page` flag resolved *before* the sidebar radio instantiates, avoiding Streamlit's `WidgetAlreadyInstantiatedError`.

> **Honest note:** there are no live scores or club fixtures — the engine is national-team and historical. Where FotMob shows "LIVE", ScoutData shows its **prediction** (🧠 chip). Live club data is a roadmap item (API-Football), not a fake.

## 🏗️ Project structure

```
ScoutData/
├── api_fastapi.py                  # REST service (contract-first, Depends + response_model)
├── app/app.py                      # Streamlit dashboard (UI + router only)
├── src/
│   ├── engine/predictor.py         # ★ pure-Python engine (no Streamlit) — shared by all fronts
│   ├── data/
│   │   ├── clean_data.py           # raw → processed
│   │   ├── create_features.py      # 18-feature engineering (leak-free)
│   │   ├── fetch_fifa_ranking.py   # ranking loader
│   │   └── update_ranking.py       # cascading self-updater (Wikipedia + ELO + fallback)
│   └── models/                     # dual XGBoost .joblib (committed → clone & run, no retrain)
├── tests/                          # 16 pytest (engine properties + API contract + convergence)
├── data/
│   ├── processed/                  # featured CSVs (committed)
│   └── external/fifa_ranking_2026.json   # hybrid ranking (versioned)
├── requirements.txt                # serving deps lock (no Streamlit, no test-deps)
├── requirements-test.txt           # pytest + httpx
├── pytest.ini                      # import-mode, testpaths, filterwarnings, max_warnings=0
└── README.md
```

## 🚀 Quick start (clone & run in 3 commands)

```bash
python -m venv venv && venv\Scripts\activate          # Windows
pip install -r requirements.txt                        # serving deps only
uvicorn api_fastapi:app --reload                       # API at :8000, docs at /docs
# or, for the dashboard:
pip install streamlit && streamlit run app/app.py
# tests:
pip install -r requirements-test.txt && pytest -v      # expect 16 passed, exit 0
```

The raw dataset (international results, 1872→2024) is from Kaggle and intentionally **not** versioned; `data/processed/` and the committed `.joblib` let you run inference without retraining. To rebuild features/models from scratch, drop the raw CSV into `data/raw/` and run `create_features.py` → `train_model.py`.

## 📊 Validation (dual-extreme, same build)

| Scenario | w_param | Result | Why it matters |
|----------|---------|--------|----------------|
| Extreme mismatch (FR vs amateur) | 100% parametric | 88.2% home | Trees can't extrapolate → formula does |
| Elite clash (AR vs ES) | 3% parametric | 41 / 28 / 30 | Dense training region → ML leads, no distortion |
| Convergence (engine == HTTP) | — | identical decimals | One brain, two mouths — verified by execution |

## ⚠️ Honest limitations

- FIFA ranking is a **current snapshot** used as a proxy for historical strength; production would use point-in-time ratings via the official API.
- Non-FIFA teams default to 700 points.
- No lineup, injury, or in-play data.
- Tree models cannot extrapolate alone — hence the parametric layer (a deliberate design choice, not a patch).
- Not yet containerized: Docker is the next infra step (the decoupled engine makes a UI-free serving image trivial once the toolchain is available).

## 🗺️ Roadmap

- [ ] Dockerize the API + run the 16 tests *inside* the image (dev/prod parity; engine already decoupled for a UI-free image).
- [ ] Club model: a second training head reusing the same hybrid ensemble, with league standings as the ranking signal (→ "predict the entire dataset", national + club).
- [ ] Live fixtures via API-Football (`/v1/jornada`), fed by a cached worker — the FotMob "LIVE" strip, honestly sourced.
- [ ] Expo/React Native client (separate repo) consuming `/v1/predict` over HTTP.

## 👤 Author

**Augusto Valentini** — Informatics Engineering student at UADE (Buenos Aires). Built as a portfolio project demonstrating production-grade Data Science: reproducible pipelines, leak-free feature engineering, hybrid modeling, contract-first serving, and honest evaluation backed by automated tests.