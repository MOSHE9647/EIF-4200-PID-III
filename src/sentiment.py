import logging
from transformers import pipeline

# Configuración básica de logs para cumplir con el manejo de errores del PID
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class SentimentAnalyzer:
    """
    Clase encargada del análisis de polaridad emocional (Sentimiento)
    para el sistema inteligente de feedback de clientes.
    """
    def __init__(self, model_name: str = "finiteautomata/beto-sentiment-analysis"):
        """
        Inicializa el pipeline de Hugging Face utilizando un modelo basado en BERT
        optimizado para español (BETO), ideal para detectar negaciones y matices.
        """
        try:
            logging.info(f"Cargando modelo de análisis de sentimiento: {model_name}...")
            self.sentiment_pipe = pipeline(
                "sentiment-analysis",
                model=model_name
            )
            self.label_mapping = {
                "POS": "POSITIVO",
                "NEU": "NEUTRO",
                "NEG": "NEGATIVO"
            }
            logging.info("Modelo de sentimiento cargado exitosamente.")
        except Exception as e:
            logging.error(f"Error al cargar el modelo de Transformers: {e}")
            raise RuntimeError(f"No se pudo inicializar el motor de sentimiento: {e}")

    def analyze_text(self, text: str) -> dict:
        """
        Analiza un único texto y devuelve su polaridad y nivel de confianza.
        Soporta validaciones de robustez para cadenas vacías o tipos incorrectos.
        """
        # Validación de robustez ante datos corruptos o vacíos (Rúbrica de Calidad)
        if not isinstance(text, str) or not text.strip():
            return {
                "label": "NEUTRO",
                "score": 0.0,
                "error": "Texto inválido o vacío"
            }

        try:
            # El modelo rinde mejor con el texto original porque evalúa el contexto completo
            raw_result = self.sentiment_pipe(text)[0]
            raw_label = raw_result['label']
            score = raw_result['score']

            # Mapeo a las etiquetas estándar solicitadas por la cátedra
            sentiment_label = self.label_mapping.get(raw_label, "NEUTRO")

            return {
                "label": sentiment_label,
                "score": round(score * 100, 2)
            }
        except Exception as e:
            logging.error(f"Error procesando el texto '{text[:30]}...': {e}")
            return {
                "label": "NEUTRO",
                "score": 0.0,
                "error": str(e)
            }

    def analyze_batch(self, texts: list) -> list:
        """
        Permite procesar colecciones de textos por lotes (Batch Processing),
        facilitando la integración con el módulo de ingesta de datos (data.py).
        """
        if not isinstance(texts, list):
            logging.error("Se esperaba una lista de textos para el procesamiento por lotes.")
            return []
            
        results = []
        for text in texts:
            results.append(self.analyze_text(text))
        return results