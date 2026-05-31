"""
Pruebas unitarias para el módulo de análisis de sentimiento (`src.sentiment`).

Estas pruebas verifican la API pública de `SentimentAnalyzer` según los
requerimientos del PID: robustez ante entradas inválidas, procesamiento
por lotes, mapeo de etiquetas y formato/escala del `score`.

Se incluyen pruebas que usan el pipeline real y una prueba aislada que
mockea el pipeline para comprobar el mapeo y el redondeo de forma
determinista (útil para CI sin descargar modelos pesados).
"""

import pytest
from src.sentiment import SentimentAnalyzer

@pytest.fixture
def analyzer():
    #Fixture que devuelve una instancia real de `SentimentAnalyzer`.
    return SentimentAnalyzer()

def test_analyze_valid_texts(analyzer):
    # Prueba básica que valida la estructura de la salida para textos válidos
    # (esperamos que sea un `dict` que contenga 'label' y 'score').
    texts = [
        "El servicio al cliente fue espectacular, resolvieron mi problema de inmediato.",
        "El producto llegó a tiempo pero la caja estaba un poco golpeada.",
        "Pésima experiencia, no vuelvo a comprar aquí, el artículo no funciona.",
        "Es bueno",
        "No es bueno",
    ]

    for text in texts:
        res = analyzer.analyze_text(text)
        assert isinstance(res, dict)
        assert 'label' in res
        assert 'score' in res
        assert isinstance(res['score'], (int, float))

def test_analyze_empty_string(analyzer):
    # Entradas vacías deben manejarse robustamente: puede devolver una
    # estructura con 'error' o una etiqueta neutra según la implementación.
    res = analyzer.analyze_text("")
    assert isinstance(res, dict)
    assert 'error' in res or 'label' in res

def test_analyze_none_input(analyzer):
    # Entradas de tipo incorrecto deben gestionarse sin romper la API.
    # La implementación actual puede lanzar TypeError/ValueError o
    # devolver un dict con clave 'error'. Ambas son aceptadas por la prueba.
    try:
        res = analyzer.analyze_text(None)
    except (TypeError, ValueError):
        return
    assert isinstance(res, dict)
    assert 'error' in res

def test_analyze_batch(analyzer):
    # Verifica que `analyze_batch` procesa listas de textos devolviendo
    # una lista con resultados (uno por texto), manteniendo la forma
    # comprobada por las pruebas anteriores.
    batch = [
        "Me encantó la comida del restaurante.",
        "El retraso en el envío arruinó todo.",
    ]
    results = analyzer.analyze_batch(batch)
    assert isinstance(results, list)
    assert len(results) == len(batch)
    for res in results:
        assert isinstance(res, dict)
        assert 'label' in res or 'error' in res

def test_label_and_score_ranges(analyzer):
    # Asegura que las etiquetas mapeadas estén dentro del conjunto
    # esperado y que el `score` esté escalado y dentro de 0..100.
    samples = [
        "Es excelente, muy recomendable.",
        "No volvería a comprarlo, muy mala calidad.",
    ]
    for text in samples:
        res = analyzer.analyze_text(text)
        assert res['label'] in ("POSITIVO", "NEUTRO", "NEGATIVO")
        assert isinstance(res['score'], (int, float))
        assert 0.0 <= res['score'] <= 100.0

def test_analyze_batch_non_list(analyzer):
    # Comportamiento ante argumentos inválidos para batch: la
    # implementación actual registra un error y devuelve lista vacía.
    res = analyzer.analyze_batch("no-es-una-lista")
    assert res == []

def test_mapping_and_rounding_with_mock():
    # Prueba determinista que mockea el pipeline para validar:
    # - el mapeo de etiquetas (POS->POSITIVO)
    # - el escalado y redondeo del score (0.87321 -> 87.32)
    #
    # Se usa `__new__` para crear la instancia sin lanzar la carga del modelo.
    class Dummy:
        def __call__(self, text):
            return [{'label': 'POS', 'score': 0.87321}]

    analyzer = SentimentAnalyzer.__new__(SentimentAnalyzer)
    # Inyectar el pipeline mockeado y el mapping esperado
    analyzer.sentiment_pipe = Dummy()
    analyzer.label_mapping = {
        "POS": "POSITIVO",
        "NEU": "NEUTRO",
        "NEG": "NEGATIVO"
    }

    out = analyzer.analyze_text("Prueba de mock")
    assert out['label'] == "POSITIVO"
    # 0.87321 * 100 = 87.321 -> redondeado a 87.32
    assert out['score'] == 87.32