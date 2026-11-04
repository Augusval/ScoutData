# src/data/update_ranking.py
"""
Actualizador automático del Ranking FIFA con merge inteligente.
Uso: python src/data/update_ranking.py

Estrategia:
  1. Top 20 oficial desde Wikipedia (puntos FIFA exactos)
  2. Merge con ranking extendido local (equipos 21-150, base ELO)
  3. Fallback completo si Wikipedia falla
"""
import json
import logging
import re
import unicodedata
import urllib.request
from pathlib import Path
from datetime import datetime
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

PROJECT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_DIR / "data" / "external" / "fifa_ranking_2026.json"
WIKI_URL = "https://en.wikipedia.org/wiki/FIFA_Men%27s_World_Ranking"

ALIAS = {
    "usa": "united states",
    "south korea": "korea republic",
    "north korea": "korea dpr",
    "iran": "ir iran",
    "turkey": "turkiye",
    "czech republic": "czechia",
    "czechia": "czechia",
    "china": "china pr",
    "ivory coast": "cote d'ivoire",
    "n ireland": "northern ireland",
    "n. ireland": "northern ireland",
    "northern ireland": "northern ireland",
    "bosnia and herzegovina": "bosnia/herz",
    "bosnia/herzegovina": "bosnia/herz",
}


def normalize(name: str) -> str:
    s = unicodedata.normalize('NFKD', str(name))
    s = ''.join(c for c in s if not unicodedata.combining(c))
    s = s.lower().replace('.', '').replace('-', ' ').strip()
    s = re.sub(r'\s+', ' ', s)
    return ALIAS.get(s, s)


def clean_cell_text(cell) -> str:
    if cell is None:
        return ""
    for sup in cell.find_all(['sup']):
        sup.decompose()
    text = cell.get_text(separator=' ', strip=True)
    text = re.sub(r'\[.*?\]', '', text).strip()
    return text


def fetch_wikipedia_top20() -> dict:
    """Extrae el Top 20 oficial de Wikipedia."""
    logging.info("[Fuente 1] Descargando Top 20 FIFA oficial desde Wikipedia...")
    req = urllib.request.Request(WIKI_URL, headers={'User-Agent': 'Mozilla/5.0 (ScoutData-portfolio)'})

    with urllib.request.urlopen(req, timeout=30) as r:
        html = r.read().decode('utf-8')

    soup = BeautifulSoup(html, 'html.parser')
    tables = soup.find_all('table')
    logging.info(f"  Analizando {len(tables)} tablas...")

    # Buscar la tabla del Top 20
    for i, table in enumerate(tables):
        headers = [clean_cell_text(th).lower() for th in table.find_all('th')]
        has_rank = any('rank' in h for h in headers)
        has_team = any('team' in h for h in headers)
        has_points = any('point' in h for h in headers)

        if has_rank and has_team and has_points:
            rows = table.find_all('tr')
            logging.info(f"  ✓ Tabla {i}: Top 20 detectada ({len(rows)} filas)")

            rank_idx = next((j for j, h in enumerate(headers) if 'rank' in h), None)
            team_idx = next((j for j, h in enumerate(headers) if 'team' in h), None)
            points_idx = next((j for j, h in enumerate(headers) if 'point' in h), None)

            ranking = {}
            for row in rows:
                cells = row.find_all(['td', 'th'])
                if len(cells) <= max(rank_idx, team_idx, points_idx):
                    continue
                try:
                    team = normalize(clean_cell_text(cells[team_idx]))
                    if not team or team == 'nan':
                        continue
                    points = float(clean_cell_text(cells[points_idx]).replace(',', '').replace(' ', ''))
                    rank = int(clean_cell_text(cells[rank_idx]).replace('=', '').replace(' ', ''))
                    ranking[team] = {"rank": rank, "points": round(points, 2)}
                except (ValueError, IndexError):
                    continue

            if len(ranking) >= 15:  # Aceptamos si tenemos al menos 15 del Top 20
                logging.info(f"  ✅ Top 20 extraído: {len(ranking)} equipos oficiales.")
                return ranking

    raise ValueError("No se encontró Top 20 en Wikipedia.")


def load_extended_ranking() -> dict:
    """Carga el ranking extendido local (equipos 21-150, base ELO)."""
    if not OUTPUT_PATH.exists():
        return {}
    with open(OUTPUT_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('ranking', {})


def merge_rankings(top20: dict, extended: dict) -> dict:
    """Combina Top 20 oficial con ranking extendido."""
    merged = extended.copy()
    overrides = 0
    for team, info in top20.items():
        if team in merged:
            overrides += 1
        merged[team] = info
    logging.info(f"  → Merge: {len(top20)} oficiales + {len(extended)} extendidos = {len(merged)} total ({overrides} overrides).")
    return merged


def update_ranking():
    """Función principal: Top 20 oficial + merge extendido."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    sources_used = []
    final_ranking = {}

    # --- Paso 1: Top 20 oficial ---
    top20 = {}
    try:
        top20 = fetch_wikipedia_top20()
        sources_used.append("Wikipedia Top 20 (FIFA oficial)")
    except Exception as e:
        logging.warning(f"  ⚠ Wikipedia falló: {e}")

    # --- Paso 2: Ranking extendido local ---
    extended = load_extended_ranking()
    if extended:
        sources_used.append(f"Ranking extendido local ({len(extended)} equipos, base ELO)")

    # --- Paso 3: Merge ---
    if top20 and extended:
        final_ranking = merge_rankings(top20, extended)
    elif top20:
        final_ranking = top20
    elif extended:
        final_ranking = extended
        sources_used.append("Solo extendido local")
    else:
        logging.error("❌ No hay datos disponibles.")
        return

    # --- Guardar ---
    data = {
        "timestamp": datetime.now().isoformat(),
        "sources": sources_used,
        "source": " + ".join(sources_used),
        "count": len(final_ranking),
        "methodology": "Top 20: puntos FIFA oficiales (Wikipedia). Equipos 21-150: rating ELO convertido a escala FIFA (0.85*ELO-300).",
        "ranking": final_ranking
    }
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logging.info(f"\n✅ Ranking actualizado: {len(final_ranking)} equipos.")
    logging.info(f"   Fuentes: {', '.join(sources_used)}")
    logging.info(f"   Metodología: {data['methodology']}")

    top5 = sorted(final_ranking.items(), key=lambda kv: kv[1]['points'], reverse=True)[:5]
    logging.info("\n🏆 Top 5:")
    for team, info in top5:
        logging.info(f"   #{info['rank']} {team}: {info['points']:.2f} pts")


if __name__ == "__main__":
    update_ranking()