"""
Pruebas unitarias para el módulo de ingesta de datos.
"""

import pytest
import json
import tempfile
from pathlib import Path
import sys
import os

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data import DataIngestion, load_comments_from_file


class TestDataIngestion:
    """Pruebas para la clase DataIngestion."""
    
    @pytest.fixture
    def temp_dir(self):
        """Crea un directorio temporal para las pruebas."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    def test_load_csv_file(self):
        """Prueba: Cargar archivo CSV."""
        file_path = "data/samples/comentarios_clientes.csv"
        ingestion = DataIngestion(file_path)
        data = ingestion.load_data()
        
        assert isinstance(data, list), "Debe retornar una lista"
        assert len(data) > 0, "La lista no debe estar vacía"
        assert all(isinstance(item, str) for item in data), "Todos los items deben ser strings"
    
    def test_load_excel_file(self):
        """Prueba: Cargar archivo Excel."""
        file_path = "data/samples/resenas_productos.xlsx"
        ingestion = DataIngestion(file_path)
        data = ingestion.load_data()
        
        assert isinstance(data, list), "Debe retornar una lista"
        assert len(data) > 0, "La lista no debe estar vacía"
        assert all(isinstance(item, str) for item in data), "Todos los items deben ser strings"
    
    def test_load_json_file(self):
        """Prueba: Cargar archivo JSON."""
        file_path = "data/samples/opiniones_empresas.json"
        ingestion = DataIngestion(file_path)
        data = ingestion.load_data()
        
        assert isinstance(data, list), "Debe retornar una lista"
        assert len(data) > 0, "La lista no debe estar vacía"
        assert all(isinstance(item, str) for item in data), "Todos los items deben ser strings"
    
    def test_load_txt_file(self):
        """Prueba: Cargar archivo TXT."""
        file_path = "data/samples/resenas_clientes.txt"
        ingestion = DataIngestion(file_path)
        data = ingestion.load_data()
        
        assert isinstance(data, list), "Debe retornar una lista"
        assert len(data) > 0, "La lista no debe estar vacía"
        assert all(isinstance(item, str) for item in data), "Todos los items deben ser strings"
    
    def test_file_not_found(self):
        """Prueba: Archivo no encontrado."""
        with pytest.raises(FileNotFoundError):
            DataIngestion("archivo_inexistente.csv")
    
    def test_unsupported_format(self, temp_dir):
        """Prueba: Formato de archivo no soportado."""
        dummy_file = temp_dir / "dummy.xyz"
        dummy_file.touch()
        
        with pytest.raises(ValueError):
            DataIngestion(str(dummy_file))
    
    def test_get_stats(self):
        """Prueba: Obtener estadísticas."""
        file_path = "data/samples/comentarios_clientes.csv"
        ingestion = DataIngestion(file_path)
        stats = ingestion.get_stats()
        
        assert "total_registros" in stats, "Stats debe tener total_registros"
        assert "promedio_caracteres" in stats, "Stats debe tener promedio_caracteres"
        assert "min_caracteres" in stats, "Stats debe tener min_caracteres"
        assert "max_caracteres" in stats, "Stats debe tener max_caracteres"
        
        assert stats["total_registros"] > 0, "Total debe ser mayor a 0"
        assert stats["promedio_caracteres"] > 0, "Promedio debe ser mayor a 0"
        assert stats["min_caracteres"] >= 0, "Mínimo debe ser >= 0"
        assert stats["max_caracteres"] >= stats["min_caracteres"], "Máximo >= Mínimo"
    
    def test_load_comments_from_file(self):
        """Prueba: Función auxiliar load_comments_from_file."""
        file_path = "data/samples/comentarios_clientes.csv"
        comments = load_comments_from_file(file_path)
        
        assert isinstance(comments, list), "Debe retornar una lista"
        assert len(comments) > 0, "La lista no debe estar vacía"
        assert all(isinstance(item, str) for item in comments), "Todos los items deben ser strings"
    
    def test_filter_empty_values(self, temp_dir):
        """Prueba: Filtrado de valores vacíos."""
        csv_file = temp_dir / "test.csv"
        csv_file.write_text("comentario\nTexto válido\n\nOtro texto\n", encoding='utf-8')
        
        ingestion = DataIngestion(str(csv_file))
        data = ingestion.load_data()
        
        # Debe haber filtrado la línea vacía
        assert len(data) == 2, "Debe tener 2 registros válidos"
    
    def test_json_with_list(self, temp_dir):
        """Prueba: JSON con lista de diccionarios."""
        json_file = temp_dir / "test.json"
        data = [
            {"id": 1, "texto": "Primer comentario"},
            {"id": 2, "comentario": "Segundo comentario"},
            {"id": 3, "opinion": "Tercer comentario"}
        ]
        json_file.write_text(json.dumps(data), encoding='utf-8')
        
        ingestion = DataIngestion(str(json_file))
        loaded = ingestion.load_data()
        
        assert len(loaded) == 3, "Debe cargar 3 comentarios"
        assert all(isinstance(item, str) for item in loaded), "Todos deben ser strings"
    
    def test_csv_auto_column_detection(self):
        """Prueba: Detección automática de columna de texto en CSV."""
        file_path = "data/samples/comentarios_clientes.csv"
        ingestion = DataIngestion(file_path)
        data = ingestion.load_data()
        
        # Debe detectar la columna 'comentario'
        assert len(data) > 0, "Debe detectar y cargar la columna"
        assert "producto" in data[0].lower() or "excelente" in data[0].lower()
    
    def test_empty_file(self, temp_dir):
        """Prueba: Archivo vacío."""
        txt_file = temp_dir / "empty.txt"
        txt_file.write_text("", encoding='utf-8')
        
        ingestion = DataIngestion(str(txt_file))
        data = ingestion.load_data()
        
        assert isinstance(data, list), "Debe retornar una lista"
        assert len(data) == 0, "Lista debe estar vacía"
    
    def test_multiple_formats_consistency(self):
        """Prueba: Carga consistente desde múltiples formatos."""
        csv_ingestion = DataIngestion("data/samples/comentarios_clientes.csv")
        csv_data = csv_ingestion.load_data()
        csv_stats = csv_ingestion.get_stats()
        
        excel_ingestion = DataIngestion("data/samples/resenas_productos.xlsx")
        excel_data = excel_ingestion.load_data()
        excel_stats = excel_ingestion.get_stats()
        
        # Ambos deben retornar datos válidos
        assert len(csv_data) > 0, "CSV debe tener datos"
        assert len(excel_data) > 0, "Excel debe tener datos"
        
        # Las estadísticas deben ser consistentes
        assert csv_stats["total_registros"] == len(csv_data)
        assert excel_stats["total_registros"] == len(excel_data)


class TestDataIngestionIntegration:
    """Pruebas de integración para el módulo de ingesta."""
    
    def test_batch_processing_simulation(self):
        """Prueba: Simulación de procesamiento por lotes."""
        file_path = "data/samples/comentarios_clientes.csv"
        ingestion = DataIngestion(file_path)
        comments = ingestion.load_data()
        
        batch_size = 3
        batches = []
        
        for i in range(0, len(comments), batch_size):
            batch = comments[i:i + batch_size]
            batches.append(batch)
        
        assert len(batches) > 0, "Debe haber al menos un lote"
        assert sum(len(b) for b in batches) == len(comments), "Suma de lotes debe ser total"
    
    def test_workflow_pipeline(self):
        """Prueba: Flujo de trabajo completo."""
        # 1. Ingesta
        file_path = "data/samples/comentarios_clientes.csv"
        ingestion = DataIngestion(file_path)
        
        # 2. Carga
        comments = ingestion.load_data()
        assert len(comments) > 0
        
        # 3. Validación
        valid_comments = [c for c in comments if len(c.strip()) > 0]
        assert len(valid_comments) == len(comments)
        
        # 4. Estadísticas
        stats = ingestion.get_stats()
        assert stats["total_registros"] > 0
        
        # 5. Preparación para PLN
        assert all(isinstance(c, str) for c in valid_comments)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
