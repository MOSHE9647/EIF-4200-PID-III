"""
Dashboard de Visualización y Métricas.

Este módulo proporciona visualizaciones dinámicas para el análisis de sentimientos:
- Nube de palabras (WordCloud) con términos más frecuentes
- Gráficos de distribución de sentimientos (barras y pastel)
- Métricas agregadas y estadísticas de análisis
"""

import logging
from typing import Optional
from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

logger = logging.getLogger(__name__)


class SentimentMetrics:
    """
    Clase para calcular métricas agregadas de análisis de sentimientos.
    """
    
    def __init__(self, results: list[dict]) -> None:
        """
        Inicializa métricas basadas en resultados de análisis.
        
        Args:
            results: Lista de diccionarios con sentimientos y scores.
        """
        self.results = results
        self.total_reviews = len(results)
        self._calculate_metrics()
    
    def _calculate_metrics(self) -> None:
        """Calcula las métricas de distribución."""
        self.sentiment_counts = Counter()
        self.sentiment_scores = {}
        
        for result in self.results:
            label = result.get("label", "NEUTRO")
            score = result.get("score", 0.0)
            
            self.sentiment_counts[label] += 1
            
            if label not in self.sentiment_scores:
                self.sentiment_scores[label] = []
            self.sentiment_scores[label].append(score)
    
    def get_distribution(self) -> dict[str, float]:
        """
        Obtiene la distribución porcentual de sentimientos.
        
        Returns:
            Diccionario con etiquetas y porcentajes.
        """
        if self.total_reviews == 0:
            return {}
        
        return {
            label: round((count / self.total_reviews) * 100, 2)
            for label, count in self.sentiment_counts.items()
        }
    
    def get_average_scores(self) -> dict[str, float]:
        """
        Obtiene los scores promedio por sentimiento.
        
        Returns:
            Diccionario con etiquetas y scores promedio.
        """
        return {
            label: round(sum(scores) / len(scores), 2)
            for label, scores in self.sentiment_scores.items()
            if scores
        }
    
    def get_summary(self) -> dict:
        """
        Obtiene resumen completo de métricas.
        
        Returns:
            Diccionario con estadísticas generales.
        """
        avg_scores = self.get_average_scores()
        
        return {
            "total_reviews": self.total_reviews,
            "distribution": self.get_distribution(),
            "average_scores": avg_scores,
            "sentiment_counts": dict(self.sentiment_counts),
        }


class DashboardVisualizer:
    """
    Clase responsable de generar visualizaciones del análisis de sentimientos.
    """
    
    def __init__(self, figsize: tuple = (15, 10), style: str = "whitegrid") -> None:
        """
        Inicializa el visualizador.
        
        Args:
            figsize: Tamaño de las figuras (ancho, alto).
            style: Estilo de seaborn para los gráficos.
        """
        self.figsize = figsize
        sns.set_style(style)
        sns.set_palette("husl")
        logger.info("Dashboard visualizador inicializado con estilo: %s", style)
    
    def plot_sentiment_distribution(
        self,
        metrics: SentimentMetrics,
        chart_type: str = "bar",
        output_path: Optional[str] = None
    ) -> None:
        """
        Genera un gráfico de distribución de sentimientos.
        
        Args:
            metrics: Instancia de SentimentMetrics con datos procesados.
            chart_type: Tipo de gráfico ("bar", "pie" o "both").
            output_path: Ruta para guardar la imagen (opcional).
        """
        if metrics.total_reviews == 0:
            logger.warning("No hay datos para visualizar.")
            return
        
        distribution = metrics.get_distribution()
        labels = list(distribution.keys())
        values = list(distribution.values())
        
        if chart_type in ["bar", "both"]:
            self._plot_bar_chart(labels, values, output_path if chart_type == "bar" else None)
        
        if chart_type in ["pie", "both"]:
            self._plot_pie_chart(labels, values, output_path if chart_type == "pie" else None)
        
        if chart_type == "both":
            self._plot_combined(labels, values, output_path)
    
    def _plot_bar_chart(
        self,
        labels: list,
        values: list,
        output_path: Optional[str] = None
    ) -> None:
        """Crea un gráfico de barras."""
        plt.figure(figsize=self.figsize)
        colors = sns.color_palette("husl", len(labels))
        bars = plt.bar(labels, values, color=colors, edgecolor="black", linewidth=1.5)
        
        # Añadir valores en las barras
        for bar, value in zip(bars, values):
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{value:.1f}%",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=11
            )
        
        plt.title("Distribución de Sentimientos", fontsize=16, fontweight="bold", pad=20)
        plt.xlabel("Sentimiento", fontsize=12, fontweight="bold")
        plt.ylabel("Porcentaje (%)", fontsize=12, fontweight="bold")
        plt.ylim(0, max(values) * 1.15)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info("Gráfico de barras guardado en: %s", output_path)
        
        plt.close()
    
    def _plot_pie_chart(
        self,
        labels: list,
        values: list,
        output_path: Optional[str] = None
    ) -> None:
        """Crea un gráfico de pastel."""
        plt.figure(figsize=(10, 8))
        colors = sns.color_palette("husl", len(labels))
        
        wedges, texts, autotexts = plt.pie(
            values,
            labels=labels,
            colors=colors,
            autopct="%1.1f%%",
            startangle=90,
            explode=[0.05] * len(labels),
            textprops={"fontsize": 11, "fontweight": "bold"}
        )
        
        # Mejorar apariencia del texto de porcentaje
        for autotext in autotexts:
            autotext.set_color("white")
            autotext.set_fontsize(12)
            autotext.set_fontweight("bold")
        
        plt.title("Distribución de Sentimientos", fontsize=16, fontweight="bold", pad=20)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info("Gráfico de pastel guardado en: %s", output_path)
        
        plt.close()
    
    def _plot_combined(
        self,
        labels: list,
        values: list,
        output_path: Optional[str] = None
    ) -> None:
        """Crea una vista combinada (barras y pastel)."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        colors = sns.color_palette("husl", len(labels))
        
        # Gráfico de barras
        bars = axes[0].bar(labels, values, color=colors, edgecolor="black", linewidth=1.5)
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[0].text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{value:.1f}%",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=11
            )
        
        axes[0].set_title("Distribución (Barras)", fontsize=14, fontweight="bold")
        axes[0].set_xlabel("Sentimiento", fontsize=11, fontweight="bold")
        axes[0].set_ylabel("Porcentaje (%)", fontsize=11, fontweight="bold")
        axes[0].set_ylim(0, max(values) * 1.15)
        
        # Gráfico de pastel
        wedges, texts, autotexts = axes[1].pie(
            values,
            labels=labels,
            colors=colors,
            autopct="%1.1f%%",
            startangle=90,
            explode=[0.05] * len(labels),
            textprops={"fontsize": 10, "fontweight": "bold"}
        )
        
        for autotext in autotexts:
            autotext.set_color("white")
            autotext.set_fontweight("bold")
        
        axes[1].set_title("Distribución (Pastel)", fontsize=14, fontweight="bold")
        
        plt.suptitle("Análisis de Distribución de Sentimientos", fontsize=16, fontweight="bold", y=1.00)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info("Gráfico combinado guardado en: %s", output_path)
        
        plt.close()
    
    def plot_wordcloud(
        self,
        lemmas: list[str],
        max_words: int = 100,
        output_path: Optional[str] = None,
        width: int = 1200,
        height: int = 600
    ) -> None:
        """
        Genera una nube de palabras con los términos más frecuentes (lematizados).
        
        Args:
            lemmas: Lista de palabras lematizadas del análisis.
            max_words: Número máximo de palabras en la nube.
            output_path: Ruta para guardar la imagen (opcional).
            width: Ancho de la nube.
            height: Alto de la nube.
        """
        if not lemmas:
            logger.warning("No hay palabras para generar la nube.")
            return
        
        # Crear texto único para la nube
        text = " ".join(lemmas)
        
        # Generar nube
        wordcloud = WordCloud(
            width=width,
            height=height,
            background_color="white",
            colormap="viridis",
            max_words=max_words,
            relative_scaling=0.5,
            min_font_size=10,
            collocations=False,
            prefer_horizontal=0.7
        ).generate(text)
        
        # Visualizar
        plt.figure(figsize=(15, 8))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title("Nube de Palabras - Términos Más Frecuentes", fontsize=16, fontweight="bold", pad=20)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info("Nube de palabras guardada en: %s", output_path)
        
        plt.close()
    
    def plot_top_lemmas(
        self,
        lemmas: list[str],
        top_n: int = 20,
        output_path: Optional[str] = None
    ) -> None:
        """
        Genera un gráfico de los lemmas más frecuentes.
        
        Args:
            lemmas: Lista de palabras lematizadas.
            top_n: Número de palabras principales a mostrar.
            output_path: Ruta para guardar la imagen (opcional).
        """
        if not lemmas:
            logger.warning("No hay palabras para visualizar.")
            return
        
        # Contar frecuencias
        lemma_counts = Counter(lemmas)
        top_lemmas = lemma_counts.most_common(top_n)
        
        if not top_lemmas:
            return
        
        # Separar etiquetas y valores
        words, frequencies = zip(*top_lemmas)
        
        # Crear gráfico
        plt.figure(figsize=(12, 8))
        colors = sns.color_palette("husl", len(words))
        bars = plt.barh(words, frequencies, color=colors, edgecolor="black", linewidth=1.2)
        
        # Añadir valores en las barras
        for bar, freq in zip(bars, frequencies):
            width = bar.get_width()
            plt.text(
                width,
                bar.get_y() + bar.get_height() / 2.0,
                f"{int(freq)}",
                ha="left",
                va="center",
                fontweight="bold",
                fontsize=10
            )
        
        plt.title(f"Top {top_n} Palabras Más Frecuentes", fontsize=14, fontweight="bold", pad=20)
        plt.xlabel("Frecuencia", fontsize=12, fontweight="bold")
        plt.ylabel("Palabra", fontsize=12, fontweight="bold")
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info("Gráfico de palabras frecuentes guardado en: %s", output_path)
        
        plt.close()
    
    def plot_sentiment_scores_distribution(
        self,
        metrics: SentimentMetrics,
        output_path: Optional[str] = None
    ) -> None:
        """
        Visualiza la distribución de scores promedio por sentimiento.
        
        Args:
            metrics: Instancia de SentimentMetrics con datos procesados.
            output_path: Ruta para guardar la imagen (opcional).
        """
        avg_scores = metrics.get_average_scores()
        
        if not avg_scores:
            logger.warning("No hay datos de scores para visualizar.")
            return
        
        labels = list(avg_scores.keys())
        scores = list(avg_scores.values())
        
        plt.figure(figsize=self.figsize)
        colors = sns.color_palette("coolwarm", len(labels))
        bars = plt.bar(labels, scores, color=colors, edgecolor="black", linewidth=1.5)
        
        # Añadir valores en las barras
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{score:.2f}",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=11
            )
        
        plt.title("Score Promedio por Sentimiento", fontsize=16, fontweight="bold", pad=20)
        plt.xlabel("Sentimiento", fontsize=12, fontweight="bold")
        plt.ylabel("Score Promedio", fontsize=12, fontweight="bold")
        plt.ylim(0, 100)
        plt.grid(axis="y", alpha=0.3, linestyle="--")
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info("Gráfico de scores guardado en: %s", output_path)
        
        plt.close()


def generate_full_dashboard(
    results: list[dict],
    lemmas: list[str],
    output_dir: Optional[str] = None
) -> None:
    """
    Genera un dashboard completo con todas las visualizaciones.
    
    Args:
        results: Lista de resultados de análisis de sentimientos.
        lemmas: Lista de palabras lematizadas del análisis.
        output_dir: Directorio para guardar las imágenes (opcional).
    """
    logger.info("Generando dashboard completo...")
    
    # Calcular métricas
    metrics = SentimentMetrics(results)
    
    # Crear visualizador
    visualizer = DashboardVisualizer()
    
    # Mostrar resumen de métricas
    summary = metrics.get_summary()
    logger.info("=" * 60)
    logger.info("RESUMEN DE ANÁLISIS DE SENTIMIENTOS")
    logger.info("=" * 60)
    logger.info("Total de reseñas procesadas: %d", summary["total_reviews"])
    logger.info("Distribución de sentimientos:")
    for sentiment, percentage in summary["distribution"].items():
        logger.info("  %s: %.2f%%", sentiment, percentage)
    logger.info("Scores promedio por sentimiento:")
    for sentiment, score in summary["average_scores"].items():
        logger.info("  %s: %.2f", sentiment, score)
    logger.info("=" * 60)
    
    # Generar visualizaciones
    output_paths = {
        "bar": f"{output_dir}/distribucion_sentimientos_barras.png" if output_dir else None,
        "pie": f"{output_dir}/distribucion_sentimientos_pastel.png" if output_dir else None,
        "wordcloud": f"{output_dir}/nube_palabras.png" if output_dir else None,
        "top_lemmas": f"{output_dir}/palabras_frecuentes.png" if output_dir else None,
        "scores": f"{output_dir}/scores_promedio.png" if output_dir else None,
    }
    
    logger.info("Generando visualizaciones...")
    
    # Gráfico de distribución de sentimientos (barras)
    visualizer.plot_sentiment_distribution(
        metrics,
        chart_type="bar",
        output_path=output_paths["bar"]
    )
    
    # Gráfico de distribución de sentimientos (pastel)
    visualizer.plot_sentiment_distribution(
        metrics,
        chart_type="pie",
        output_path=output_paths["pie"]
    )
    
    # Nube de palabras
    visualizer.plot_wordcloud(
        lemmas,
        output_path=output_paths["wordcloud"]
    )
    
    # Palabras más frecuentes
    visualizer.plot_top_lemmas(
        lemmas,
        output_path=output_paths["top_lemmas"]
    )
    
    # Scores promedio
    visualizer.plot_sentiment_scores_distribution(
        metrics,
        output_path=output_paths["scores"]
    )
    
    logger.info("Dashboard completamente generado.")


__all__ = [
    "SentimentMetrics",
    "DashboardVisualizer",
    "generate_full_dashboard",
]
