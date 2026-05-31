"""
Sistema Inteligente de Análisis de Feedback y Sentimiento del Cliente
Punto de entrada principal de la aplicación (Orquestador)
"""

import logging
import sys
from pathlib import Path
import click

# Agregar src al path para importar módulos
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data import DataIngestion
from pipeline import EngineeringPipeline
from sentiment import SentimentAnalyzer


def setup_logging() -> None:
    """Configura el sistema de logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )


@click.group()
def cli() -> None:
    """Sistema de Análisis de Sentimiento y Feedback de Clientes."""
    setup_logging()


@cli.command()
@click.option(
    '--file',
    '-f',
    type=click.Path(exists=True),
    help='Ruta del archivo a procesar (CSV, Excel, JSON, TXT)'
)
def ingest(file: str) -> None:
    """
    Ingesta y carga datos desde archivos.
    
    Soporta: CSV, Excel (.xlsx), JSON, TXT
    """
    logger = logging.getLogger(__name__)
    
    if not file:
        click.echo("\n Archivos disponibles en data/samples/:")
        data_dir = Path("data/samples")
        if data_dir.exists():
            files = list(data_dir.glob("*"))
            if files:
                for i, f in enumerate(files, 1):
                    click.echo(f"  {i}. {f.name}")
            else:
                click.echo("  No hay archivos de muestra")
        
        file = click.prompt("\n Ingrese la ruta del archivo")
    
    try:
        # Crear instancia de ingesta
        ingestion = DataIngestion(file)
        
        # Cargar datos
        click.echo("\n Procesando archivo...")
        data = ingestion.load_data()
        
        # Mostrar estadísticas
        stats = ingestion.get_stats()
        click.echo("\n" + "="*60)
        click.echo(" ESTADÍSTICAS DE DATOS CARGADOS")
        click.echo("="*60)
        click.echo(f"✓ Total de registros: {stats['total_registros']}")
        click.echo(f"✓ Promedio de caracteres: {stats['promedio_caracteres']}")
        click.echo(f"✓ Mín. caracteres: {stats['min_caracteres']}")
        click.echo(f"✓ Máx. caracteres: {stats['max_caracteres']}")
        click.echo("="*60 + "\n")
        
        # Mostrar primeros registros
        if data:
            click.echo(" Primeros 3 registros:")
            for i, text in enumerate(data[:3], 1):
                preview = text[:100] + "..." if len(text) > 100 else text
                click.echo(f"  {i}. {preview}")
        
        click.echo("\n Datos cargados exitosamente")
        
    except (FileNotFoundError, ValueError) as e:
        click.echo(f"\n Error: {str(e)}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"\n Error inesperado: {str(e)}", err=True)
        logger.exception("Error en la ingesta de datos")
        sys.exit(1)


@cli.command()
def info() -> None:
    """Muestra información del sistema y formatos soportados."""
    click.echo("\n" + "="*60)
    click.echo("  INFORMACIÓN DEL SISTEMA")
    click.echo("="*60)
    click.echo("Sistema Inteligente de Análisis de Feedback")
    click.echo("\n Formatos soportados:")
    click.echo("  • CSV (.csv)")
    click.echo("  • Excel (.xlsx)")
    click.echo("  • JSON (.json)")
    click.echo("  • Texto (.txt)")
    click.echo("\n Ubicación de datos:")
    click.echo("  Coloque los archivos en: data/samples/")
    click.echo("\n Uso:")
    click.echo("  python main.py ingest --file <ruta_archivo>")
    click.echo("="*60 + "\n")


@cli.command()
def validate() -> None:
    """Valida archivos en la carpeta data/samples/."""
    logger = logging.getLogger(__name__)
    data_dir = Path("data/samples")
    
    if not data_dir.exists():
        click.echo(f" Carpeta no encontrada: {data_dir}")
        return
    
    valid_extensions = {'.csv', '.xlsx', '.json', '.txt'}
    files = [f for f in data_dir.iterdir() if f.suffix.lower() in valid_extensions]
    
    if not files:
        click.echo(f"  No hay archivos válidos en {data_dir}")
        return
    
    click.echo("\n" + "="*60)
    click.echo(" VALIDACIÓN DE ARCHIVOS")
    click.echo("="*60)
    
    for file_path in files:
        try:
            ingestion = DataIngestion(file_path)
            data = ingestion.load_data()
            click.echo(f"✓ {file_path.name}: {len(data)} registros")
        except Exception as e:
            click.echo(f"✗ {file_path.name}: {str(e)}")
    
    click.echo("="*60 + "\n")


@cli.command()
@click.option(
    '--file',
    '-f',
    type=click.Path(exists=True),
    help='Ruta del archivo a procesar (CSV, Excel, JSON, TXT)'
)
@click.option(
    '--show-processed',
    '-s',
    is_flag=True,
    help='Mostrar textos procesados por el pipeline'
)
def process(file: str, show_processed: bool) -> None:
    """
    Flujo completo: Ingesta → Pipeline (Normalización) → Análisis de Sentimiento.
    
    Procesa comentarios desde el inicio hasta obtener su polaridad emocional.
    """
    logger = logging.getLogger(__name__)
    
    if not file:
        click.echo("\n Archivos disponibles en data/samples/:")
        data_dir = Path("data/samples")
        if data_dir.exists():
            files = list(data_dir.glob("*"))
            if files:
                for i, f in enumerate(files, 1):
                    click.echo(f"  {i}. {f.name}")
            else:
                click.echo("  No hay archivos de muestra")
        
        file = click.prompt("\n Ingrese la ruta del archivo")
    
    try:
        # PASO 1: INGESTA DE DATOS
        click.echo("\n" + "="*70)
        click.echo("📥 PASO 1: INGESTA DE DATOS")
        click.echo("="*70)
        
        ingestion = DataIngestion(file)
        raw_comments = ingestion.load_data()
        stats = ingestion.get_stats()
        
        click.echo(f"✓ Archivo: {Path(file).name}")
        click.echo(f"✓ Registros cargados: {stats['total_registros']}")
        click.echo(f"✓ Promedio caracteres: {stats['promedio_caracteres']}")
        
        # PASO 2: PIPELINE DE NORMALIZACIÓN
        click.echo("\n" + "="*70)
        click.echo(" PASO 2: PIPELINE DE NORMALIZACIÓN")
        click.echo("="*70)
        
        pipeline = EngineeringPipeline()
        click.echo(" Procesando textos (limpieza, tokenización, lematización)...")
        
        processed_results = []
        for i, comment in enumerate(raw_comments, 1):
            result = pipeline.process_text(comment)
            processed_results.append(result)
            if i % max(1, len(raw_comments) // 5) == 0:
                click.echo(f"  {i}/{len(raw_comments)} comentarios procesados...")
        
        click.echo(f"✓ Procesamiento completado")
        click.echo(f"✓ Tokens totales extraídos: {sum(r.token_count for r in processed_results)}")
        
        # Mostrar ejemplos si se solicita
        if show_processed and processed_results:
            click.echo("\n Ejemplo de procesamiento (primer comentario):")
            ex = processed_results[0]
            click.echo(f"  Original: {ex.original_text[:80]}...")
            click.echo(f"  Limpio:   {ex.clean_text[:80]}...")
            click.echo(f"  Tokens:   {ex.tokens[:5]}...")
            click.echo(f"  Lemas:    {ex.lemmas[:5]}...")
        
        # PASO 3: ANÁLISIS DE SENTIMIENTO
        click.echo("\n" + "="*70)
        click.echo(" PASO 3: ANÁLISIS DE SENTIMIENTO")
        click.echo("="*70)
        
        sentiment_analyzer = SentimentAnalyzer()
        click.echo(" Analizando polaridad emocional...")
        
        # Usar los textos procesados (lematizados)
        processed_texts = [r.processed_text for r in processed_results]
        sentiments = sentiment_analyzer.analyze_batch(processed_texts)
        
        click.echo("✓ Análisis completado")
        
        # RESULTADOS FINALES
        click.echo("\n" + "="*70)
        click.echo(" RESULTADOS FINALES")
        click.echo("="*70)
        
        # Contar sentimientos
        sentiment_counts = {"POSITIVO": 0, "NEUTRO": 0, "NEGATIVO": 0}
        total_score = 0
        
        for sentiment in sentiments:
            label = sentiment.get("label", "NEUTRO")
            score = sentiment.get("score", 0)
            sentiment_counts[label] = sentiment_counts.get(label, 0) + 1
            total_score += score
        
        # Mostrar estadísticas
        total = len(sentiments)
        click.echo(f"\nDistribución de Sentimientos:")
        click.echo(f"   POSITIVO:  {sentiment_counts['POSITIVO']:3d} ({sentiment_counts['POSITIVO']*100/total:5.1f}%)")
        click.echo(f"   NEUTRO:    {sentiment_counts['NEUTRO']:3d} ({sentiment_counts['NEUTRO']*100/total:5.1f}%)")
        click.echo(f"   NEGATIVO:  {sentiment_counts['NEGATIVO']:3d} ({sentiment_counts['NEGATIVO']*100/total:5.1f}%)")
        
        avg_score = total_score / total if total > 0 else 0
        click.echo(f"\nConfianza Promedio: {avg_score:.2f}%")
        
        # Mostrar detalle de algunos resultados
        click.echo(f"\n Primeros 5 Análisis Detallados:")
        for i, (comment, sentiment) in enumerate(zip(raw_comments[:5], sentiments[:5]), 1):
            preview = comment[:60] + "..." if len(comment) > 60 else comment
            label = sentiment.get("label", "NEUTRO")
            score = sentiment.get("score", 0)
            
            emoji = "😊" if label == "POSITIVO" else "😞" if label == "NEGATIVO" else "😐"
            click.echo(f"\n  {i}. {emoji} [{label}] ({score}%)")
            click.echo(f"     \"{preview}\"")
        
        click.echo("\n" + "="*70)
        click.echo(" PROCESAMIENTO COMPLETADO EXITOSAMENTE")
        click.echo("="*70 + "\n")
        
    except (FileNotFoundError, ValueError) as e:
        click.echo(f"\n Error: {str(e)}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"\n Error inesperado: {str(e)}", err=True)
        logger.exception("Error en el flujo de procesamiento")
        sys.exit(1)



def main() -> None:
    """Punto de entrada de la aplicación."""
    cli()


if __name__ == "__main__":
    main()
