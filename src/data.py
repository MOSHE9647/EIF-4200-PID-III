"""
Módulo de ingesta de datos para el análisis de sentimiento.
Soporta múltiples formatos: CSV, Excel, JSON, TXT.
"""

import logging
from pathlib import Path
from typing import List, Dict, Union, Optional
import pandas as pd
import json


class DataIngestion:
    """Clase para la ingesta y validación de datos desde múltiples fuentes."""

    SUPPORTED_FORMATS = {'.csv', '.xlsx', '.json', '.txt'}
    
    def __init__(self, file_path: Union[str, Path]) -> None:
        """
        Inicializa la ingesta de datos.
        
        Args:
            file_path: Ruta del archivo a procesar
            
        Raises:
            FileNotFoundError: Si el archivo no existe
            ValueError: Si el formato no es soportado
        """
        self.file_path = Path(file_path)
        self.logger = logging.getLogger(__name__)
        
        if not self.file_path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {self.file_path}")
        
        if self.file_path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Formato no soportado: {self.file_path.suffix}. "
                f"Formatos válidos: {self.SUPPORTED_FORMATS}"
            )
    
    def load_data(self) -> List[str]:
        """
        Carga los datos del archivo según su formato.
        
        Returns:
            Lista de textos (comentarios/reseñas)
            
        Raises:
            ValueError: Si hay errores al procesar el archivo
        """
        file_ext = self.file_path.suffix.lower()
        
        self.logger.info(f"Cargando datos desde: {self.file_path}")
        
        try:
            if file_ext == '.csv':
                data = self._load_csv()
            elif file_ext == '.xlsx':
                data = self._load_excel()
            elif file_ext == '.json':
                data = self._load_json()
            elif file_ext == '.txt':
                data = self._load_txt()
            
            self.logger.info(f"✓ Se cargaron {len(data)} registros exitosamente")
            return data
            
        except Exception as e:
            self.logger.error(f"Error al cargar datos: {str(e)}")
            raise
    
    def _load_csv(self) -> List[str]:
        """Carga datos desde archivo CSV."""
        df = pd.read_csv(self.file_path)
        self.logger.info(f"CSV cargado con columnas: {list(df.columns)}")
        
        # Buscar columna con textos (puede ser 'comentario', 'texto', 'resena', etc.)
        text_column = self._find_text_column(df)
        
        if text_column is None:
            raise ValueError(
                f"No se encontró columna de texto. Columnas disponibles: {list(df.columns)}"
            )
        
        # Filtrar valores nulos y convertir a string
        texts = df[text_column].dropna().astype(str).tolist()
        return [text.strip() for text in texts if text.strip()]
    
    def _load_excel(self) -> List[str]:
        """Carga datos desde archivo Excel."""
        df = pd.read_excel(self.file_path)
        self.logger.info(f"Excel cargado con columnas: {list(df.columns)}")
        
        # Buscar columna con textos
        text_column = self._find_text_column(df)
        
        if text_column is None:
            raise ValueError(
                f"No se encontró columna de texto. Columnas disponibles: {list(df.columns)}"
            )
        
        # Filtrar valores nulos y convertir a string
        texts = df[text_column].dropna().astype(str).tolist()
        return [text.strip() for text in texts if text.strip()]
    
    def _load_json(self) -> List[str]:
        """Carga datos desde archivo JSON."""
        with open(self.file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Si es una lista de diccionarios
        if isinstance(data, list):
            texts = []
            for item in data:
                if isinstance(item, dict):
                    # Buscar campo de texto
                    text = self._extract_text_from_dict(item)
                    if text:
                        texts.append(text)
                elif isinstance(item, str):
                    texts.append(item)
            return texts
        
        # Si es un diccionario
        elif isinstance(data, dict):
            text = self._extract_text_from_dict(data)
            return [text] if text else []
        
        else:
            raise ValueError(f"Formato JSON no válido: {type(data)}")
    
    def _load_txt(self) -> List[str]:
        """Carga datos desde archivo TXT."""
        with open(self.file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Separar por saltos de línea o por párrafos
        texts = [line.strip() for line in content.split('\n') if line.strip()]
        return texts
    
    @staticmethod
    def _find_text_column(df: pd.DataFrame) -> Optional[str]:
        """
        Identifica automáticamente la columna que contiene textos/comentarios.
        
        Args:
            df: DataFrame de pandas
            
        Returns:
            Nombre de la columna encontrada o None
        """
        # Palabras clave comunes en nombres de columnas
        keywords = ['comentario', 'texto', 'resena', 'review', 'comment', 
                    'opinion', 'feedback', 'descripcion', 'content', 'texto_original']
        
        columns_lower = {col.lower(): col for col in df.columns}
        
        # Buscar coincidencias exactas o parciales
        for keyword in keywords:
            for col_lower, col_original in columns_lower.items():
                if keyword in col_lower:
                    return col_original
        
        # Si no encuentra, retorna la última columna (convención común)
        if len(df.columns) > 0:
            return df.columns[-1]
        
        return None
    
    @staticmethod
    def _extract_text_from_dict(item: dict) -> Optional[str]:
        """
        Extrae texto de un diccionario JSON.
        
        Args:
            item: Diccionario del JSON
            
        Returns:
            Texto encontrado o None
        """
        keywords = ['comentario', 'texto', 'resena', 'review', 'comment', 
                    'opinion', 'feedback', 'descripcion', 'content']
        
        for key in item.keys():
            if key.lower() in keywords:
                value = item[key]
                if isinstance(value, str):
                    return value.strip()
        
        # Si no encuentra, retorna el primer valor string
        for value in item.values():
            if isinstance(value, str):
                return value.strip()
        
        return None
    
    def get_stats(self) -> Dict[str, Union[int, float]]:
        """
        Retorna estadísticas del conjunto de datos cargado.
        
        Returns:
            Diccionario con estadísticas
        """
        data = self.load_data()
        
        if not data:
            return {"total_registros": 0, "promedio_caracteres": 0}
        
        char_counts = [len(text) for text in data]
        
        return {
            "total_registros": len(data),
            "promedio_caracteres": round(sum(char_counts) / len(char_counts), 2),
            "min_caracteres": min(char_counts),
            "max_caracteres": max(char_counts)
        }


def load_comments_from_file(file_path: Union[str, Path]) -> List[str]:
    """
    Función auxiliar para cargar comentarios desde un archivo.
    
    Args:
        file_path: Ruta del archivo
        
    Returns:
        Lista de comentarios
    """
    ingestion = DataIngestion(file_path)
    return ingestion.load_data()
