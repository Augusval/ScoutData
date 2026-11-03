# src/analysis/generate_eda.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path

# Configuración de Logs para ver el progreso
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Rutas automáticas
PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_DIR / "data" / "processed" / "matches_clean.csv"
IMAGES_DIR = PROJECT_DIR / "reports" / "images"

# Crear carpeta de imágenes si no existe
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# Estilo profesional de gráficos
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

def main():
    logging.info("Cargando datos procesados...")
    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    
    logging.info("Generando Gráfico 1: Ventaja de Local...")
    # Gráfico 1: Ventaja de Local
    result_counts = df['home_win'].value_counts()
    labels = ['No (Empate/Visita)', 'Sí (Gana Local)']
    sizes = [result_counts.get(0, 0), result_counts.get(1, 0)]
    
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff'])
    plt.title('¿Existe ventaja de jugar de local? (Histórico global)', fontweight='bold')
    plt.savefig(IMAGES_DIR / "01_ventaja_local.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info("Generando Gráfico 2: Evolución de Goles...")
    # Gráfico 2: Evolución de goles (desde 1990)
    goals_per_year = df[df['year'] >= 1990].groupby('year')['total_goals'].mean().reset_index()
    
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=goals_per_year, x='year', y='total_goals', marker='o', color='b', linewidth=2)
    plt.title('Evolución del promedio de goles por partido (1990 - Presente)', fontweight='bold')
    plt.xlabel('Año')
    plt.ylabel('Promedio de Goles Totales')
    plt.xticks(rotation=45)
    plt.savefig(IMAGES_DIR / "02_evolucion_goles.png", dpi=300, bbox_inches='tight')
    plt.close()

    logging.info("Generando Gráfico 3: Top Torneos...")
    # Gráfico 3: Top Torneos
    tournament_stats = df.groupby('tournament').agg(
        total_matches=('date', 'count'),
        avg_goals=('total_goals', 'mean')
    ).reset_index()
    top_tournaments = tournament_stats[tournament_stats['total_matches'] >= 100].nlargest(5, 'avg_goals')
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=top_tournaments, x='avg_goals', y='tournament', palette='viridis')
    plt.title('Top 5 Torneos con mayor promedio de goles (Min. 100 partidos)', fontweight='bold')
    plt.xlabel('Promedio de Goles')
    plt.savefig(IMAGES_DIR / "03_top_torneos.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"✅ ¡ÉXITO! Gráficos guardados en alta calidad en: {IMAGES_DIR}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Error al generar EDA: {e}")