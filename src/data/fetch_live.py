# src/data/fetch_live.py
"""
Fetch de fixtures EN VIVO desde API-Football (api-sports.io).
Replica la experiencia FotMob: tira EN VIVO global + agrupada por liga.
Diseno defensivo: key por env, cache con TTL, fallback si no hay key/red/quota.
Usa solo stdlib (urllib) -> no agrega dependencias al lock de serving.

FIXES APLICADOS:
  FIX A  _norm_fixture recibe el ITEM COMPLETO de response (no el sub-dict
         'fixture'); mato la doble anidacion que daba 86 fixtures vacios.
  FIX B  (superado por FIX D) season ya no se manda al board; ver FIX D.
  FIX C  get_live_board detecta parseo sospechosamente vacio (ratio home=='?')
         y lo loggea + marca en meta, para que un parseo roto no sea invisible.
  FIX D  PIVOT FREE PLAN: la API-Sports free tier SOLO accede a seasons
         2022-2024, por eso '/fixtures?league=&season=2026' da error de plan.
         Pero '/fixtures?live=all' SI funciona (live es siempre 'ahora').
         Entonces el board agrupa los live POR LIGA EN MEMORIA (filtrando por
         league_id), sin tocar el endpoint paywalled. Se pierde la grilla de
         'programados no iniciados' (NS) de hoy; se gana la tira roja en vivo
         agrupada por liga, que es el corazon de FotMob.
  FIX E  Se elimina el loop today-por-liga del board automatico (antes quemaba
         5 requests en llamadas que SIEMPRE fallan en free con season actual).
         Board = 1 request (live=all). La grilla historica (season 2022-2024)
         queda opt-in via '--grid lid season date' para screenshot de README.

Uso CLI:
  python src/data/fetch_live.py                       # board (1 request)
  python src/data/fetch_live.py --discover AR         # ids de liga por pais
  python src/data/fetch_live.py --grid 128 2024 2024-11-10   # grilla historica (opt-in)
"""
import os
import json
import time
import logging
import argparse
import urllib.request
import urllib.parse
import urllib.error
from pathlib import Path
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_DIR / "data" / "external" / "live_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

BASE = "https://v3.football.api-sports.io"
TZ = "America/Argentina/Buenos_Aires"
TTL_LIVE = 60          # segundos
TTL_GRID = 900         # segundos (grilla historica opt-in)
# FIX E: default verificado via --discover (antes 133=Primera D estaba mal
# etiquetado como Serie A). 39=PL, 140=LaLiga, 135=Serie A, 61=Ligue 1,
# 128=Liga Profesional AR. El usuario setea SCOUTDATA_LEAGUE_IDS igual.
DEFAULT_LEAGUES = [39, 140, 135, 61, 128]

LIVE_STATUSES = {"1H", "2H", "HT", "ET", "BT", "P", "AET", "PEN"}
FINISHED_STATUSES = {"FT", "AET", "PEN"}
PRE_STATUSES = {"NS", "TBD", "HTE", "CANCELLED", "POSTPONED", "ABANDONED", "INTERRUPTED"}


class LiveError(Exception):
    """Error de fuente con motivo LEGIBLE (no_key / quota / net_* / api_error: <detalle>)."""
    pass


def _key() -> str:
    return os.environ.get("SCOUTDATA_API_KEY", "").strip()


def _league_ids() -> list:
    raw = os.environ.get("SCOUTDATA_LEAGUE_IDS", "").strip()
    if not raw:
        return DEFAULT_LEAGUES
    return [int(x) for x in raw.split(",") if x.strip().isdigit()]


def _cache_path(kind: str) -> Path:
    return CACHE_DIR / f"{kind}.json"


def _cache_get(kind: str, ttl: int):
    p = _cache_path(kind)
    if not p.exists():
        return None
    try:
        blob = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    if time.time() - blob.get("_ts", 0) > ttl:
        return None
    return blob.get("data")


def _cache_set(kind: str, data):
    p = _cache_path(kind)
    p.write_text(json.dumps({"_ts": time.time(), "data": data}, ensure_ascii=False), encoding="utf-8")


def _api_get(path: str, params: dict) -> dict:
    """GET con key; levanta LiveError con motivo legible ante fallo/quota.

    API-Sports puede devolver HTTP 200 con errors en el body, por eso se revisa
    payload['errors'] ADEMAS del status code. Los errors NUNCA se descartan:
    se loggean y se transportan en el LiveError (Fix 2 de turnos previos).
    """
    qs = urllib.parse.urlencode(params)
    url = f"{BASE}{path}?{qs}"
    req = urllib.request.Request(url, headers={
        "x-apisports-key": _key(),
        "User-Agent": "ScoutData-portfolio",
    })
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            payload = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8")[:300]
        except Exception:
            pass
        if e.code == 429 or "quota" in body.lower() or "limit" in body.lower():
            raise LiveError("quota")
        raise LiveError(f"http_{e.code}: {body}")
    except Exception as e:
        raise LiveError(f"net_{type(e).__name__}: {e}")

    errs = payload.get("errors") or {}
    if errs:
        logger.error("API errors en %s: %s", path, errs)
        msg = str(errs).lower()
        if "quota" in msg or "limit" in msg:
            raise LiveError("quota")
        raise LiveError(f"api_error: {errs}")
    return payload


def _norm_fixture(item: dict) -> dict:
    """FIX A: recibe el ITEM COMPLETO de response (no el sub-dict 'fixture').

    Estructura real de API-Sports v3, cada item de response:
        { "fixture": {...}, "league": {...}, "teams": {...}, "goals": {...}, "venue": {...} }
    Por eso leemos item['fixture'], item['league'], etc. AL MISMO nivel.
    """
    fix = item.get("fixture", {})
    lg = item.get("league", {})
    th = item.get("teams", {}).get("home", {})
    ta = item.get("teams", {}).get("away", {})
    gl = item.get("goals", {})
    st = fix.get("status", {})
    date_iso = fix.get("date") or ""
    country = lg.get("country", "")
    if isinstance(country, dict):          # algunas respuestas traen country como dict
        country = country.get("name", "")
    return {
        "league_id": lg.get("id"),
        "league": lg.get("name", "?"),
        "country": country,
        "home": th.get("name", "?"),
        "away": ta.get("name", "?"),
        "home_id": th.get("id"),
        "away_id": ta.get("id"),
        "hs": gl.get("home"),
        "as": gl.get("away"),
        "status": st.get("short", "TBD"),
        "elapsed": st.get("elapsed"),
        "kick_time": date_iso[11:16],
        "kick_date": date_iso[:10],
        "venue": (fix.get("venue") or {}).get("name", ""),
    }


def _empty_ratio(rows: list) -> float:
    """FIX C: fraccion de fixtures parseados con home=='?' (sintoma de parseo roto).

    Distingue 'cero partidos' (rows==[], ratio 0, legitimo) de 'recibi N items
    pero estan vacios' (ratio alto, regresion de _norm_fixture).
    """
    if not rows:
        return 0.0
    empties = sum(1 for r in rows if r.get("home") == "?")
    return empties / len(rows)


def fetch_global_live() -> list:
    """Tira EN VIVO global (1 request, todas las competiciones). /fixtures SI acepta timezone."""
    cached = _cache_get("global_live", TTL_LIVE)
    if cached is not None:
        return cached
    payload = _api_get("/fixtures", {"live": "all", "timezone": TZ})
    out = [_norm_fixture(f) for f in payload.get("response", [])]   # FIX A: f, no f["fixture"]
    _cache_set("global_live", out)
    return out


def fetch_historical_grid(lid: int, season: int, date: str) -> list:
    """FIX E (opt-in): grilla de una liga/temporada/fecha. season DEBE estar en
    2022-2024 para free tier; si no, la API da error de plan (transparente via
    Fix 2). NO se usa en el board automatico; es para screenshot de README.
    """
    cached = _cache_get(f"grid_{lid}_{season}_{date}", TTL_GRID)
    if cached is not None:
        return cached
    payload = _api_get("/fixtures", {
        "league": lid, "season": season, "date": date, "timezone": TZ,
    })
    out = [_norm_fixture(f) for f in payload.get("response", [])]   # FIX A
    _cache_set(f"grid_{lid}_{season}_{date}", out)
    return out


def get_live_board() -> dict:
    """
    Board estilo FotMob (FIX D + FIX E):
      - live: todos los partidos pitando en el mundo (1 request, live=all).
      - live_by_league: los live AGRUPADOS por liga, filtrando por league_id en
        memoria (0 requests extra). Reemplaza la grilla 'hoy por liga' que en
        free tier esta paywalled (season actual no accesible).
      - meta: incluye free_plan_note documentando la limitacion honesta.
    Nunca crashea: sin key / sin red / quota -> devuelve vacio + motivo en meta.
    """
    meta = {
        "fetched_at": datetime.now().isoformat(timespec="seconds"),
        "source": "API-Football",
        "degraded": None,
        "leagues": _league_ids(),
        "free_plan_note": (
            "live=all funciona en free (agrupado por liga en memoria). "
            "La grilla de 'programados no iniciados' (NS) de HOY requiere "
            "season 2026, que esta behind paywall en free (solo 2022-2024). "
            "Usa --grid lid 2024 <fecha> para una jornada historica de demo."
        ),
    }
    if not _key():
        meta["degraded"] = "no_key"
        return {"live": [], "live_by_league": {}, "meta": meta}

    try:
        live = fetch_global_live()                       # FIX E: 1 request, no 6
        er = _empty_ratio(live)                          # FIX C
        if er > 0.5:
            logger.warning("Parseo sospechoso en live: %.0f%% de %d fixtures con "
                           "campos vacios (posible regresion de _norm_fixture)",
                           er * 100, len(live))
            meta["parse_suspicious_live"] = round(er, 2)
    except LiveError as e:
        meta["degraded"] = f"live:{e}"
        live = []

    # FIX D: agrupar EN MEMORIA por liga (0 requests extra), filtrando por LEAGUE_IDS
    wanted = set(_league_ids())
    groups: dict[str, list] = {}
    for f in live:
        if f.get("league_id") in wanted:
            key = f'{f.get("country", "")} - {f.get("league", "?")}'
            groups.setdefault(key, []).append(f)

    return {"live": live, "live_by_league": groups, "meta": meta}


def discover_league(query: str) -> None:
    """Imprime candidatos de liga por pais o nombre para hallar el id correcto.

    /leagues NO acepta 'timezone' (es param de /fixtures); mandarlo generaba el
    'api_error' mudo de turnos atras. Se omitio. El print del country se limpia
    (antes mostraba el dict completo; ahora solo el nombre).
    """
    if not _key():
        print("[ERROR] Setea SCOUTDATA_API_KEY primero.")
        return
    params = {}
    if query.strip().lower() in {"argentina", "ar", "arg"}:
        params["country"] = "Argentina"
    else:
        params["search"] = query
    try:
        payload = _api_get("/leagues", params)
    except LiveError as e:
        print(f"[ERROR] {e}")
        return
    rows = payload.get("response", [])
    if not rows:
        print(f"Sin resultados para '{query}'. Probe el nombre exacto del pais.")
        return
    print(f"\n[BUSCAR] Candidatos para '{query}':")
    for item in rows:
        lg = item.get("league", {})
        ctry = item.get("country", "")
        if isinstance(ctry, dict):
            ctry = ctry.get("name", "")
        print(f"   id={lg.get('id')}  |  {lg.get('name')}  ({ctry})")
    print("\nCopia el id y sumalo a SCOUTDATA_LEAGUE_IDS.\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--discover", nargs="?", const="Argentina", default=None)
    ap.add_argument("--grid", nargs=3, metavar=("LID", "SEASON", "DATE"),
                    help="grilla historica opt-in (season 2022-2024 para free), ej: --grid 128 2024 2024-11-10")
    args = ap.parse_args()
    if args.discover:
        discover_league(args.discover)
    elif args.grid:
        lid, season, date = args.grid
        try:
            rows = fetch_historical_grid(int(lid), int(season), date)
        except LiveError as e:
            print(f"[ERROR] {e}")
            raise SystemExit(1)
        print(f"\nGrilla historica league={lid} season={season} date={date}: {len(rows)} partidos")
        for f in rows[:10]:
            print(f"  {f['home']} {f['hs']}-{f['as']} {f['away']}  ({f['status']})  [{f['league']}]")
    else:
        b = get_live_board()
        print(f"\nmeta={b['meta']}")
        print(f"EN VIVO global: {len(b['live'])}")
        if b["live_by_league"]:
            print("EN VIVO por liga (filtrado de live=all, 0 requests extra):")
            for k, v in b["live_by_league"].items():
                print(f"  {k}: {len(v)} en vivo")
        else:
            print("(ninguna liga de SCOUTDATA_LEAGUE_IDS tiene partidos pitando en este momento)")
        for i, f in enumerate(b["live"][:2]):
            print(f"  live[{i}]: {f['home']} {f['hs']}-{f['as']} {f['away']}  "
                  f"({f['status']} {f['elapsed']}')  [{f['league']}]")