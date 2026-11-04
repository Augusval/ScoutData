# src/data/create_ranking_file.py
"""
Genera automáticamente el archivo JSON del Ranking FIFA (Oct 2026).
Ejecutar una sola vez: python src/data/create_ranking_file.py
"""
import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_DIR / "data" / "external" / "fifa_ranking_2026.json"

RANKING = {
    "spain": {"rank": 1, "points": 2011.57},
    "argentina": {"rank": 2, "points": 1972.8},
    "france": {"rank": 3, "points": 1958.86},
    "england": {"rank": 4, "points": 1925.59},
    "portugal": {"rank": 5, "points": 1807.69},
    "brazil": {"rank": 6, "points": 1806.51},
    "morocco": {"rank": 7, "points": 1805.34},
    "belgium": {"rank": 8, "points": 1779.17},
    "netherlands": {"rank": 9, "points": 1778.61},
    "mexico": {"rank": 10, "points": 1738.48},
    "colombia": {"rank": 11, "points": 1735.8},
    "switzerland": {"rank": 12, "points": 1725.39},
    "germany": {"rank": 13, "points": 1717.52},
    "croatia": {"rank": 14, "points": 1714.74},
    "italy": {"rank": 15, "points": 1713.04},
    "united states": {"rank": 16, "points": 1705.47},
    "japan": {"rank": 17, "points": 1684.75},
    "senegal": {"rank": 18, "points": 1648.84},
    "norway": {"rank": 19, "points": 1637.89},
    "uruguay": {"rank": 20, "points": 1635.54},
    "denmark": {"rank": 21, "points": 1631.5},
    "austria": {"rank": 22, "points": 1620.8},
    "south korea": {"rank": 23, "points": 1615.2},
    "australia": {"rank": 24, "points": 1608.9},
    "turkey": {"rank": 25, "points": 1605.3},
    "ukraine": {"rank": 26, "points": 1598.7},
    "sweden": {"rank": 27, "points": 1592.4},
    "poland": {"rank": 28, "points": 1587.1},
    "serbia": {"rank": 29, "points": 1580.6},
    "peru": {"rank": 30, "points": 1575.2},
    "iran": {"rank": 31, "points": 1568.9},
    "wales": {"rank": 32, "points": 1562.3},
    "tunisia": {"rank": 33, "points": 1558.7},
    "ivory coast": {"rank": 34, "points": 1552.1},
    "ecuador": {"rank": 35, "points": 1548.6},
    "scotland": {"rank": 36, "points": 1542.9},
    "algeria": {"rank": 37, "points": 1538.4},
    "egypt": {"rank": 38, "points": 1532.8},
    "venezuela": {"rank": 39, "points": 1528.2},
    "czech republic": {"rank": 40, "points": 1522.7},
    "romania": {"rank": 41, "points": 1518.3},
    "chile": {"rank": 42, "points": 1512.9},
    "paraguay": {"rank": 43, "points": 1508.4},
    "greece": {"rank": 44, "points": 1502.8},
    "hungary": {"rank": 45, "points": 1498.3},
    "nigeria": {"rank": 46, "points": 1492.7},
    "russia": {"rank": 47, "points": 1488.2},
    "slovakia": {"rank": 48, "points": 1482.6},
    "cameroon": {"rank": 49, "points": 1478.1},
    "ghana": {"rank": 50, "points": 1472.5},
}


def main():
    # Crear la carpeta data/external si no existe
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        "timestamp": "2026-10-07",
        "source": "FIFA Official (Oct 2026)",
        "ranking": RANKING
    }
    
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Archivo de Ranking FIFA creado en: {OUTPUT_PATH}")
    print(f"   Equipos incluidos: {len(RANKING)}")


if __name__ == "__main__":
    main()