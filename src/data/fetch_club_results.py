# src/data/fetch_club_results.py
"""
Descarga resultados HISTORICOS de clubes desde football-data.co.uk (CSVs por
liga por temporada, gratis, sin key, sin quota). Fuente de training para el
modelo de clubes (la segunda cabeza del Hybrid Ensemble).

Por que football-data.co.uk y NO API-Football: el free tier de API-Football
solo accede a seasons 2022-2024 (paywall ya pagado en el en vivo), insuficiente
para training largo. football-data.co.uk da 10+ temporadas gratis, HTTP plain.

Formato CSV: Date, HomeTeam, AwayTeam, FTHG, FTAG (+ odds que ignoro).
Date varia por temporada: 'dd/mm/yy' (antiguas) vs 'dd/mm/yyyy' (recientes).
El parser maneja ambos asumiendo siglo correcto para el rango 2014-2024.

Guarda un CSV unificado en data/raw/club_results.csv (NO se commitea: el
.gitignore excluye data/raw/, igual que results.csv de selecciones; el README
documenta 'corre fetch_club_results.py para regenerar').

Uso:
  python src/data/fetch_club_results.py
  python src/data/fetch_club_results.py --start 2014 --end 2024
"""
import io
import sys
import time
import logging
import argparse
import urllib.request
import urllib.error
from datetime import datetime

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# code -> nombre legible. Codes oficiales de football-data.co.uk.
LEAGUES = {
    "E0": "Premier League",
    "SP1": "La Liga",
    "I1": "Serie A",
    "D1": "Bundesliga",
    "F1": "Ligue 1",
}
BASE_URL = "https://www.football-data.co.uk/mmz4281"
COLS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
SLEEP_BETWEEN = 0.3          # buen ciudadano HTTP (no hay quota, pero respeto)
TIMEOUT = 30


def _season_code(start: int, end: int) -> str:
    """2014,2015 -> '1415' (pattern de la URL de football-data.co.uk)."""
    return f"{start % 100:02d}{end % 100:02d}"


def _parse_date(tok: str):
    """Parser robusto para 'dd/mm/yy' Y 'dd/mm/yyyy'.

    football-data.co.uk mezcla formatos segun temporada. Asumo siglo correcto
    para el rango 2014-2024: anio de 2 digitos -> 2000+yy (todos los yy del
    rango 14-24 caen en 2014-2024, safe). Si falla, devuelvo None (la fila se
    droppea despues, no crashea el batch).
    """
    if not tok or pd.isna(tok):
        return None
    parts = str(tok).strip().split("/")
    if len(parts) != 3:
        return None
    d, m, y = parts
    try:
        yi = int(y)
        if len(y) == 2:
            yi = 2000 + yi          # rango 2014-2024: yy 14-24 -> 2014-2024
        return datetime(int(yi), int(m), int(d))
    except (ValueError, TypeError):
        return None


def _download_one(code: str, season: str) -> pd.DataFrame | None:
    """Descarga y parsea UN CSV (liga+temporada). None si 404/fallo (skip, no crash)."""
    url = f"{BASE_URL}/{season}/{code}.csv"
    req = urllib.request.Request(url, headers={"User-Agent": "ScoutData-portfolio"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            raw = r.read()
    except urllib.error.HTTPError as e:
        logger.warning("HTTP %s en %s/%s (skip)", e.code, season, code)
        return None
    except Exception as e:
        logger.warning("net_%s en %s/%s (skip)", type(e).__name__, season, code)
        return None

    try:
        # usecols -> solo las 5 que importan (ignora odds, evita columnas raras)
        df = pd.read_csv(io.BytesIO(raw), usecols=lambda c: c in COLS,
                         encoding="latin-1")   # football-data usa latin-1 (acentos en equipos)
    except Exception as e:
        logger.warning("parse fallo en %s/%s: %s (skip)", season, code, e)
        return None

    missing = [c for c in COLS if c not in df.columns]
    if missing:
        logger.warning("%s/%s sin columnas %s (skip)", season, code, missing)
        return None

    df["Date"] = df["Date"].map(_parse_date)
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    df["FTHG"] = pd.to_numeric(df["FTHG"], errors="coerce")
    df["FTAG"] = pd.to_numeric(df["FTAG"], errors="coerce")
    df = df.dropna(subset=["FTHG", "FTAG"])
    df["FTHG"] = df["FTHG"].astype(int)
    df["FTAG"] = df["FTAG"].astype(int)
    df["league_code"] = code
    df["league"] = LEAGUES[code]
    df["season"] = season
    return df[COLS + ["league_code", "league", "season"]]


def fetch_all(start: int, end: int) -> pd.DataFrame:
    """Baja todas las ligas x temporadas [start, end). Acumula, no crashea por 1 fallo."""
    frames, failures = [], []
    seasons = [(y, y + 1) for y in range(start, end)]
    total = len(seasons) * len(LEAGUES)
    done = 0
    for (s, e) in seasons:
        code = _season_code(s, e)
        for lc in LEAGUES:
            done += 1
            logger.info("[%d/%d] %s %s (%s)", done, total, lc, LEAGUES[lc], code)
            df = _download_one(lc, code)
            if df is None or df.empty:
                failures.append(f"{lc}/{code}")
                continue
            frames.append(df)
            time.sleep(SLEEP_BETWEEN)
    if not frames:
        logger.error("NINGUN CSV bajo sanos. Fallos: %s", failures)
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values("Date").reset_index(drop=True)
    logger.info("Descargados %d filas sanas. Fallos (skip): %d -> %s",
                len(out), len(failures), failures[:10])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2014, help="temporada inicial (inicio)")
    ap.add_argument("--end", type=int, default=2024, help="temporada final (inicio, exclusivo)")
    args = ap.parse_args()

    df = fetch_all(args.start, args.end)
    if df.empty:
        print("[ERROR] no hay datos. Revisá red / rango de temporadas.")
        sys.exit(1)

    out_path = "data/raw/club_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\n=== RESUMEN club_results.csv ===")
    print(f"filas totales: {len(df)}")
    print(f"rango fechas: {df['Date'].min().date()} -> {df['Date'].max().date()}")
    print(f"ligas: {sorted(df['league'].unique())}")
    print(f"temporadas: {sorted(df['season'].unique())}")
    print(f"equipos únicos: {len(set(df['HomeTeam']) | set(df['AwayTeam']))}")
    print(f"NaN goles restantes (debe ser 0): {df[['FTHG','FTAG']].isna().sum().sum()}")
    print(f"\nHead:\n{df.head(3).to_string(index=False)}")
    print(f"\nPor liga:\n{df.groupby('league').size().to_string()}")
    print(f"\nGuardado en {out_path} (NO se commitea: .gitignore excluye data/raw/)")


if __name__ == "__main__":
    main()