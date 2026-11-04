# src/data/fetch_fifa_ranking.py
"""
Ranking FIFA masculino desde archivo JSON local (actualizado Oct 2026).
Funciona 100% garantizado sin depender de internet.
"""
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

PROJECT_DIR = Path(__file__).resolve().parents[2]
RANKING_FILE = PROJECT_DIR / "data" / "external" / "fifa_ranking_2026.json"
DEFAULT_POINTS = 700

ALIAS = {
    "usa": "united states",
    "south korea": "korea republic",
    "north korea": "korea dpr",
    "iran": "ir iran",
    "turkey": "turkiye",
    "czech republic": "czechia",
    "china": "china pr",
    "ivory coast": "cote d'ivoire",
}


def normalize(name: str) -> str:
    """Normaliza nombres para comparar."""
    import unicodedata
    s = unicodedata.normalize('NFKD', str(name))
    s = ''.join(c for c in s if not unicodedata.combining(c))
    s = s.lower().replace('.', '').strip()
    return ALIAS.get(s, s)


def get_fifa_ranking() -> dict:
    """Carga el ranking FIFA desde el archivo JSON local."""
    if not RANKING_FILE.exists():
        logging.error(f"Archivo de ranking no encontrado: {RANKING_FILE}")
        return {}
    
    with open(RANKING_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    ranking = data.get('ranking', {})
    logging.info(f"✅ Ranking FIFA cargado: {len(ranking)} equipos (actualizado: {data.get('timestamp', 'N/A')})")
    return ranking


def get_team_fifa(team_name: str, ranking: dict) -> dict:
    """Devuelve {rank, points} de un equipo."""
    entry = ranking.get(normalize(team_name))
    if entry is None:
        return {"rank": None, "points": DEFAULT_POINTS}
    return entry


if __name__ == "__main__":
    ranking = get_fifa_ranking()
    print(f"\nEquipos rankeados: {len(ranking)}")
    
    if len(ranking) > 0:
        print("\nTop 10:")
        top10 = sorted(ranking.items(), key=lambda kv: kv[1]['points'], reverse=True)[:10]
        for team, info in top10:
            print(f"  #{info['rank']} {team}: {info['points']:.2f} pts")
        print(f"\nFrancia: {get_team_fifa('France', ranking)}")
        print(f"Argentina: {get_team_fifa('Argentina', ranking)}")
        print(f"España: {get_team_fifa('Spain', ranking)}")
        print(f"Arameans Suryoye (sin ranking): {get_team_fifa('Arameans Suryoye', ranking)}")
    else:
        print("\n⚠️ No se pudo cargar el ranking FIFA.")