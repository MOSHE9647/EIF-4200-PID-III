"""
Sistema Inteligente de Análisis de Feedback y Sentimiento del Cliente
Punto de entrada principal de la aplicación (Orquestador CLI)
"""

import logging
import subprocess
import sys
from pathlib import Path
import click

# Agregamos src al path para que los módulos internos se descubran correctamente.
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data import DataIngestion
from pipeline import EngineeringPipeline
from sentiment import SentimentAnalyzer

def setup_logging() -> None:
    """Configura el sistema de logging para el orquestador."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

def _list_sample_files() -> None:
    """Muestra los archivos de ejemplo disponibles en data/samples/."""
    click.echo("\n[INFO] Archivos disponibles en data/samples/:")
    data_dir = Path("data/samples")
    if data_dir.exists():
        files = list(data_dir.glob("*"))
        if files:
            for index, file_path in enumerate(files, 1):
                click.echo(f"  {index}. {file_path.name}")
            return
    click.echo("  No hay archivos de muestra en el directorio.")

def _resolve_input_file(file_path: str | None) -> str:
    """Resuelve la ruta del archivo; si no se provee, pide seleccionar interactivamente."""
    if file_path:
        return file_path
    
    _list_sample_files()
    data_dir = Path("data/samples")
    if not data_dir.exists() or not list(data_dir.glob("*")):
        click.echo("\n[ERROR] Se requiere proveer un archivo con --file")
        sys.exit(1)

    choice = click.prompt("\nSelecciona el número del archivo a procesar", type=int)
    files = list(data_dir.glob("*"))
    if 1 <= choice <= len(files):
        return str(files[choice - 1])
        
    click.echo("\n[ERROR] Selección inválida")
    sys.exit(1)

# ==========================================
# DEFINICIÓN DEL GRUPO DE COMANDOS CLI
# ==========================================

@click.group()
def cli() -> None:
    """Orquestador Principal del Sistema de Análisis PLN."""
    setup_logging()

@cli.command()
def dashboard() -> None:
    """Lanza la interfaz de usuario interactiva (Streamlit)."""
    click.echo("\n[INFO] Iniciando Dashboard Analítico Visual...")
    click.echo("La interfaz se abrirá en tu navegador predeterminado.")
    
    dashboard_path = Path(__file__).parent / "src" / "dashboard.py"
    
    if not dashboard_path.exists():
        click.echo(f"\n[ERROR] No se encontró el archivo del dashboard en {dashboard_path}")
        return
        
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(dashboard_path)])
    except KeyboardInterrupt:
        click.echo("\n[INFO] Dashboard detenido por el usuario.")
    except Exception as e:
        click.echo(f"\n[ERROR] Falla crítica al ejecutar el dashboard: {e}")

@cli.command()
@click.option('--file', '-f', type=click.Path(exists=True), help='Ruta del archivo a analizar.')
def ingest(file: str) -> None:
    """Extrae datos y muestra métricas sin analizarlos lingüísticamente."""
    target_file = _resolve_input_file(file)
    click.echo(f"\n[INFO] Iniciando proceso de ingesta para: {target_file}")
    
    try:
        ingestion = DataIngestion(target_file)
        stats = ingestion.get_stats()
        
        click.echo("\nEstadísticas del Conjunto de Datos:")
        click.echo("-" * 40)
        click.echo(f" Total de registros: {stats.get('total_registros', 0)}")
        click.echo(f" Promedio de caracteres: {stats.get('promedio_caracteres', 0)}")
        if 'min_caracteres' in stats and 'max_caracteres' in stats:
            click.echo(f" Rango de longitud: {stats['min_caracteres']} a {stats['max_caracteres']} caracteres")
        
        data = ingestion.load_data()
        if data:
            click.echo("\nExtracto de los primeros registros:")
            for i, text in enumerate(data[:3], 1):
                preview = text[:80] + "..." if len(text) > 80 else text
                click.echo(f"  {i}. {preview}")
                
        click.echo("\n[INFO] Ingesta completada exitosamente.")
    except Exception as e:
        click.echo(f"\n[ERROR] Falla durante la ingesta: {e}")

@cli.command()
def validate() -> None:
    """Verifica que los archivos de muestra puedan leerse correctamente."""
    click.echo("\n[INFO] Validando integridad de los datos de muestra...")
    data_dir = Path("data/samples")
    
    if not data_dir.exists():
        click.echo("  [ERROR] Directorio data/samples/ no encontrado.")
        return
        
    files = list(data_dir.glob("*"))
    if not files:
        click.echo("  [INFO] No hay archivos en data/samples/ para validar.")
        return
        
    click.echo(f"Se encontraron {len(files)} archivos.\n")
    
    for file_path in files:
        try:
            ingestion = DataIngestion(file_path)
            stats = ingestion.get_stats()
            click.echo(f"  [OK] {file_path.name} - {stats.get('total_registros', 0)} registros listos.")
        except Exception as e:
            click.echo(f"  [ERROR] {file_path.name} - Falla en validación: {e}")

@cli.command()
@click.option('--file', '-f', type=click.Path(exists=True), help='Ruta del archivo a analizar.')
@click.option('--show-processed', is_flag=True, help='Muestra el texto en consola después de la lematización.')
def process(file: str, show_processed: bool) -> None:
    """Ejecuta el pipeline de NLP e inferencia completo por terminal."""
    target_file = _resolve_input_file(file)
    
    click.echo("\n" + "=" * 60)
    click.echo("  INICIANDO PIPELINE DE PROCESAMIENTO NLP")
    click.echo("=" * 60)
    
    try:
        # FASE 1
        click.echo("\n[1/3] Extrayendo y limpiando datos de origen...")
        ingestion = DataIngestion(target_file)
        raw_texts = ingestion.load_data()
        click.echo(f"  ✓ {len(raw_texts)} registros cargados.")
        
        # FASE 2
        click.echo("\n[2/3] Aplicando normalización lingüística (spaCy)...")
        pipeline = EngineeringPipeline()
        processed_texts = []
        for text in raw_texts:
            result = pipeline.process_text(text)
            if result and result.clean_text:
                processed_texts.append(result.clean_text)
                if show_processed:
                    click.echo(f"    Original: {text[:40]}...")
                    click.echo(f"    Lematizado: {result.clean_text[:40]}...\n")
                    
        click.echo(f"  ✓ {len(processed_texts)} registros estructuralmente válidos.")
        
        if not processed_texts:
            click.echo("\n[ERROR] Ningún texto sobrevivió a la limpieza semántica.")
            return
            
        # FASE 3
        click.echo("\n[3/3] Evaluando sentimiento (Modelo Transformers)...")
        analyzer = SentimentAnalyzer()
        results = analyzer.analyze_batch(processed_texts)
        
        positives = sum(1 for r in results if r.get('label') == 'POSITIVO')
        negatives = sum(1 for r in results if r.get('label') == 'NEGATIVO')
        neutrals = sum(1 for r in results if r.get('label') == 'NEUTRO')
        
        click.echo("\n  RESUMEN EJECUTIVO DE TERMINAL:")
        click.echo("  ------------------------------")
        click.echo(f"  Volumen total evaluado: {len(results)}")
        click.echo(f"  😊 Positivos:  {positives}")
        click.echo(f"  😐 Neutros:    {neutrals}")
        click.echo(f"  😞 Negativos:  {negatives}")
        
    except Exception as e:
        click.echo(f"\n[ERROR] Falla crítica durante la ejecución en cadena: {e}")

@cli.command()
def info() -> None:
    """Muestra la configuración y dependencias del ecosistema."""
    click.echo("\n" + "=" * 60)
    click.echo("  INFORMACIÓN TÉCNICA DEL SISTEMA")
    click.echo("=" * 60)
    click.echo("\nPlataforma de Auditoría de PLN e Inteligencia Artificial")
    
    click.echo("\n Formatos Nativos Soportados:")
    click.echo("  • CSV (.csv)    - Ideal para exportaciones de bases de datos")
    click.echo("  • Excel (.xlsx) - Reportes corporativos")
    click.echo("  • JSON (.json)  - Datos estructurados provenientes de APIs")
    click.echo("  • Texto (.txt)  - Registros en texto plano directo")
    
    click.echo("\n Ubicación de Carga de Lotes:")
    click.echo("  Directorio absoluto: data/samples/")
    
    click.echo("\n Listado de Comandos Disponibles:")
    click.echo("  python main.py dashboard")
    click.echo("  python main.py process --file <ruta_archivo>")
    click.echo("  python main.py process --file <ruta_archivo> --show-processed")
    click.echo("  python main.py ingest --file <ruta_archivo>")
    click.echo("  python main.py validate")
    click.echo("  python main.py --help")
    click.echo("\n" + "=" * 60)

if __name__ == '__main__':
    cli()