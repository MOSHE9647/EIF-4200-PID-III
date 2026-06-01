"""
Dashboard interactivo que carga datos REALES desde data/samples/
Integración completa: DataIngestion → Pipeline → SentimentAnalyzer → Visualizaciones
"""

import matplotlib
matplotlib.use('Agg')  # Backend sin GUI
import matplotlib.pyplot as plt

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from data import DataIngestion
from pipeline import EngineeringPipeline
from sentiment import SentimentAnalyzer
from dashboard import generate_full_dashboard


def setup_logging():
    """Configura logging básico."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )


def find_data_files():
    """Busca archivos en data/samples/."""
    data_dir = Path("data/samples")
    
    if not data_dir.exists():
        print(f" Carpeta no encontrada: {data_dir}")
        return []
    
    # Buscar archivos soportados
    supported_ext = {'.csv', '.xlsx', '.json', '.txt'}
    files = [f for f in data_dir.iterdir() if f.suffix.lower() in supported_ext]
    
    if not files:
        print(f" No hay archivos en {data_dir}")
        return []
    
    return files


def load_all_data(files: list) -> list:
    """Carga datos de múltiples archivos."""
    all_texts = []
    
    for file_path in files:
        try:
            print(f"\n Cargando: {file_path.name}")
            ingestion = DataIngestion(file_path)
            texts = ingestion.load_data()
            all_texts.extend(texts)
            print(f"    {len(texts)} registros cargados")
        except Exception as e:
            print(f"    Error al cargar {file_path.name}: {e}")
            continue
    
    return all_texts


def main():
    """Proceso completo de análisis."""
    
    setup_logging()
    
    print("\n" + "="*70)
    print("DASHBOARD DE ANÁLISIS DE SENTIMIENTOS - DATOS REALES")
    print("="*70)
    
    # Paso 1: Buscar archivos
    print("\n  BUSCANDO ARCHIVOS EN data/samples/...")
    files = find_data_files()
    
    if not files:
        print("   No hay archivos para procesar.")
        return
    
    print(f"    Se encontraron {len(files)} archivo(s)")
    for file in files:
        print(f"     • {file.name}")
    
    # Paso 2: Cargar datos
    print("\n  CARGANDO DATOS...")
    all_texts = load_all_data(files)
    
    if not all_texts:
        print("    No se cargaron datos.")
        return
    
    print(f"\n    TOTAL DE TEXTOS: {len(all_texts)}")
    
    # Paso 3: Inicializar componentes
    print("\n  INICIALIZANDO COMPONENTES...")
    try:
        print("   • Cargando modelo de sentimiento...")
        analyzer = SentimentAnalyzer()
        
        print("   • Cargando pipeline de procesamiento...")
        pipeline = EngineeringPipeline()
        print("    Componentes listos")
    except Exception as e:
        print(f"    Error al inicializar: {e}")
        return
    
    # Paso 4: Procesar textos
    print("\n  PROCESANDO TEXTOS CON NLP...")
    try:
        processed_texts = pipeline.process_batch(all_texts)
        
        # Extraer lemmas
        all_lemmas = []
        for processed in processed_texts:
            all_lemmas.extend(processed['lemmas'])
        
        print(f"   ✓ Procesados {len(processed_texts)} textos")
        print(f"   ✓ Se extrajeron {len(all_lemmas)} lemmas")
        
    except Exception as e:
        print(f"   Error en procesamiento: {e}")
        return
    
    # Paso 5: Analizar sentimientos
    print("\n  ANALIZANDO SENTIMIENTOS...")
    try:
        sentiment_results = analyzer.analyze_batch(all_texts)
        
        # Contar sentimientos
        sentiments = {}
        for result in sentiment_results:
            label = result.get('label', 'NEUTRO')
            sentiments[label] = sentiments.get(label, 0) + 1
        
        print("   Análisis completado. Distribución:")
        for sentiment, count in sorted(sentiments.items()):
            percentage = (count / len(sentiment_results)) * 100
            print(f"     • {sentiment}: {count} ({percentage:.1f}%)")
            
    except Exception as e:
        print(f"   Error en análisis: {e}")
        return
    
    # Paso 6: Generar visualizaciones
    print("\n  GENERANDO VISUALIZACIONES...")
    try:
        output_dir = Path("outputs/dashboard_datos_reales")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        generate_full_dashboard(
            results=sentiment_results,
            lemmas=all_lemmas,
            output_dir=str(output_dir)
        )
        
        # Listar archivos generados
        print(f"\n   Dashboard generado en: {output_dir}")
        print("\n   ARCHIVOS GENERADOS:")
        for file in sorted(output_dir.glob("*.png")):
            size_mb = file.stat().st_size / (1024*1024)
            print(f"     • {file.name} ({size_mb:.2f} MB)")
        
    except Exception as e:
        print(f"  Error generando visualizaciones: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "="*70)
    print("PROCESO COMPLETADO EXITOSAMENTE")
    print("="*70)
    print(f"\nResumen:")
    print(f"  • Archivos procesados: {len(files)}")
    print(f"  • Textos analizados: {len(all_texts)}")
    print(f"  • Lemmas extraídos: {len(all_lemmas)}")
    print(f"  • Visualizaciones: 5 gráficos PNG")
    print(f"\nRuta de salida: {output_dir}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProceso interrumpido por el usuario")
    except Exception as e:
        print(f"\nError fatal: {e}")
        import traceback
        traceback.print_exc()
