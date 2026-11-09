# ⚽ ScoutData — Hybrid Football Prediction Engine (ML + REST API + Live + Clubs)

End-to-end football prediction system with **two model heads**:

- **National teams**: FIFA/ELO hybrid ensemble for international matches.
- **Clubs**: intra-league ELO hybrid ensemble for major European domestic leagues.

Both heads share the same transparent architecture: dual XGBoost expected-goals regressors + parametric rating-difference calibration + Poisson probability layer. Shipped as a **multi-page Streamlit dashboard**, a **contract-first FastAPI service**, and a **live match feed** powered by API-Football.

![Python](https://img.shields.io/badge/Python-3.15-blue)
![XGBoost](https://img.shields.io/badge/ML-XGBoost-orange)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![pytest](https://img.shields.io/badge/tests-30%20passed-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🎯 The problem

Naïve football predictors fail in two ways:

1. **Unrealistic extremes** — binary classifiers cannot extrapolate mismatches like France vs. a non-FIFA side.
2. **Domain blindness** — a national-team model should not pretend to understand club football, and a club model should not fake cross-league comparisons when the source cannot support them.

ScoutData solves this with a **hybrid ensemble** and **explicit domain boundaries**:

- National matches: FIFA/ELO rating difference + ML.
- Club matches: intra-league ELO + ML.
- Unknown or non-comparable matchups: honest `None` / HTTP 404 / UI caption, never an invented number.

---

## 🧬 Architecture: one ensemble, two heads

Every prediction fuses two independent signals:

| Layer | National head | Club head |
|---|---|---|
| ML regressors | Dual XGBoost for home/away xG | Dual XGBoost for home/away club xG |
| Parametric signal | FIFA/ELO rating difference | Intra-league ELO difference |
| Probability layer | Poisson goal model | Poisson goal model |
| Blend gate | Magnitude of rating gap | Magnitude of ELO gap |

The blend weight grows when the matchup becomes extreme:

```text
advantage = |rating_home - rating_away| / divisor
w_param   = min(1.0, advantage / 3.0)
xG_final  = (1 - w_param) * xG_ML + w_param * xG_parametric
```

Calibration:

- National head uses FIFA-point scale, divisor around 400.
- Club head uses intra-league ELO scale, divisor 150, derived from the observed ELO distribution:
  - national extreme ≈ 1259 FIFA points → 1259 / 400 ≈ 3.15
  - club extreme ≈ 460 ELO points → 460 / 150 ≈ 3.07

This preserves the same parametric behavior across domains without reusing an ill-scaled constant.

---

## 🔌 One engine, several fronts

All domain logic is framework-agnostic:

- `src/engine/predictor.py` — national engine, pure Python.
- `src/engine/predictor_clubs.py` — club engine, pure Python.
- `src/data/fetch_live.py` — live fixture fetcher, stdlib only.
- `src/data/fetch_club_results.py` — historical club results downloader.
- `src/data/create_club_features.py` — point-in-time club feature engineering.
- `src/data/train_club_model.py` — dual XGBoost club training.

These modules are consumed by:

- the **Streamlit dashboard** (`app/app.py`);
- the **FastAPI service** (`api_fastapi.py`);
- the **pytest suite** (`tests/`).

A dedicated convergence test proves the engine and HTTP endpoint return identical numbers for both heads.

---

## 🌐 REST API

### National teams

```bash
curl -X POST http://127.0.0.1:8000/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"home":"France","away":"Arameans Suryoye"}'
```

Returns typed JSON with:

- `xg.ml`, `xg.parametric`, `xg.final`
- `blend.parametric_pct`
- `probabilities_pct`
- `most_likely_score`
- ranking/domain metadata

Errors:

- unknown team → `404`
- same team → `422`
- missing field → `422`

### Clubs

```bash
curl -X POST http://127.0.0.1:8000/v1/predict/club \
  -H "Content-Type: application/json" \
  -d '{"home":"Manchester City","away":"Norwich"}'
```

The club endpoint accepts API-Football-style names and canonicalizes them internally:

```text
Manchester City -> Man City
Paris Saint-Germain -> Paris SG
Atletico Madrid -> Ath Madrid
```

Errors:

- cross-league / unavailable pair → `404`
- same team after canonicalization → `422`
- missing field → `422`

Example honest failure:

```bash
curl -X POST http://127.0.0.1:8000/v1/predict/club \
  -H "Content-Type: application/json" \
  -d '{"home":"Manchester City","away":"Barcelona"}'
```

Returns `404`, because the club source is intra-league only and cannot compare Premier League vs. La Liga without a European-competition bridge.

Interactive OpenAPI docs:

```text
http://127.0.0.1:8000/docs
```

---

## 🔴 Live match feed

The **Jornada** page shows a live strip powered by API-Football:

- currently live matches;
- score and minute;
- grouped by followed league;
- graceful degradation when no key/quota/calendar window is available.

Setup:

```powershell
$env:SCOUTDATA_API_KEY = "YOUR_API_FOOTBALL_KEY"
$env:SCOUTDATA_LEAGUE_IDS = "128,1032,39,140,15"
```

Discover league IDs:

```powershell
python src/data/fetch_live.py --discover Argentina
```

Free-tier limitation:

- `live=all` works.
- Historical/current-day league grids requiring `season=2026` are behind the free paywall.
- Therefore the UI shows matches already kicking off, not all scheduled “NS” matches for the day.

This limitation is documented in the UI and README, not hidden.

---

## 🏟️ Club model

The club head uses **football-data.co.uk** historical results:

- Premier League
- La Liga
- Serie A
- Bundesliga
- Ligue 1
- 2014/15 → 2023/24
- 18,085 matches
- 160 canonical club names

Pipeline:

```powershell
python src/data/fetch_club_results.py
python src/data/create_club_features.py
python src/data/train_club_model.py
```

Outputs:

```text
data/raw/club_results.csv                         # not committed
data/processed/club_matches_with_features.csv     # committed
src/models/xgboost_club_home.joblib               # committed
src/models/xgboost_club_away.joblib               # committed
```

Club features are point-in-time:

- ELO updated after each match;
- rolling form from previous matches only;
- Bayesian strength shrinkage;
- experience ratio;
- no future leakage.

Club domain boundary:

- ✅ intra-league European club matches present in history;
- ❌ cross-league matches;
- ❌ Champions/Europa League;
- ❌ Argentine club league;
- ❌ live 2026 ELO recalculation.

For a live 2026 club match, the current implementation uses the latest historical encounter as a point-in-time proxy. This is explicitly labeled as a limitation.

---

## 🧪 Tests

```powershell
pip install -r requirements-test.txt
pytest -v
```

Expected:

```text
30 passed
```

Coverage:

- national engine properties;
- club engine properties;
- national API contract;
- club API contract;
- alias normalization;
- cross-league 404 behavior;
- same-team 422 behavior;
- response-model filtering;
- engine/API convergence for both heads.

`pytest.ini` enforces a zero-warning ratchet:

```ini
max_warnings = 0
```

Known third-party warnings are filtered with anchored regexes, not silenced globally.

---

## 🏗️ Project structure

```text
ScoutData/
├── api_fastapi.py                      # FastAPI service, two heads
├── app/app.py                          # Streamlit dashboard
├── src/
│   ├── engine/
│   │   ├── predictor.py                # national engine, pure Python
│   │   └── predictor_clubs.py          # club engine, pure Python
│   ├── data/
│   │   ├── fetch_live.py               # API-Football live fetcher, stdlib only
│   │   ├── fetch_club_results.py       # football-data.co.uk downloader
│   │   ├── create_club_features.py     # point-in-time club features
│   │   ├── train_club_model.py         # dual XGBoost club training
│   │   └── smoke_clubs.py              # club engine smoke test
│   └── models/
│       ├── xgboost_home_goals.joblib
│       ├── xgboost_away_goals.joblib
│       ├── xgboost_club_home.joblib
│       └── xgboost_club_away.joblib
├── tests/
│   ├── test_engine.py
│   ├── test_api.py
│   └── test_clubs.py
├── data/
│   ├── processed/
│   │   ├── matches_with_features.csv
│   │   └── club_matches_with_features.csv
│   └── external/
│       └── fifa_ranking_2026.json
├── requirements.txt
├── requirements-test.txt
├── pytest.ini
├── .gitignore
└── README.md
```

---

## 🚀 Quick start

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Run dashboard:

```powershell
streamlit run app/app.py
```

Run API:

```powershell
uvicorn api_fastapi:app --reload
```

Run tests:

```powershell
pip install -r requirements-test.txt
pytest -v
```

The **Clubes** page works without an API key because it uses local committed models and processed features. The live strip requires `SCOUTDATA_API_KEY`.

---

## 📊 Validation

| Scenario | Head | Result | Why it matters |
|---|---|---|---|
| France vs. amateur | National | ~100% parametric, high home win | Trees cannot extrapolate; formula does |
| Argentina vs. Spain | National | low parametric weight, balanced 1X2 | ML leads in dense training region |
| Man City vs. Norwich | Club | high ELO gap, high parametric weight | Club extreme handled correctly |
| Arsenal vs. Chelsea | Club | moderate ELO gap, mixed blend | No artificial certainty in close games |
| Man City vs. Barcelona | Club | `None` / HTTP 404 | Cross-league is outside source domain |
| Engine vs. HTTP | Both | identical decimals | One brain, multiple mouths |

---

## ⚠️ Honest limitations

- National FIFA ranking is a current snapshot, not point-in-time for every historical match.
- Non-FIFA teams default to 700 points.
- No lineups, injuries, weather, or in-play variables.
- Tree models cannot extrapolate alone; the parametric layer is a deliberate design choice.
- Live free tier does not expose current-season scheduled grids; only currently live matches are shown.
- Club model is intra-league only.
- Club model does not cover Champions/Europa League or Argentine club football.
- Club live predictions use the latest historical encounter as a proxy, not a live 2026 ELO recalculation.
- Not yet containerized; Docker is a roadmap item.

---

## 🗺️ Roadmap

- [ ] Dockerize API and run tests inside the image.
- [ ] Incremental ELO worker for live club 2026 updates.
- [ ] Paid API-Football tier or alternate source for current-season scheduled grids.
- [ ] Argentine club source and second club regional head.
- [ ] Expo/React Native mobile client consuming `/v1/predict` and `/v1/predict/club`.

---

## 👤 Author

**Augusto Valentini** — Informatics Engineering student at UADE, Buenos Aires.

Built as a portfolio project demonstrating production-grade Data Science and Software Engineering: reproducible pipelines, leak-free feature engineering, hybrid modeling, contract-first serving, domain-aware limitations, automated tests, and publishable repository hygiene.