"""
Plataforma de Auditoría Analítica - Dashboard de Visualización Avanzado
Implementa un diseño de "Tema Universal Agnóstico" que responde automáticamente 
a la configuración nativa de Streamlit (Claro/Oscuro).
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
from pathlib import Path
import sys
import logging

# Configuración de rutas e importaciones internas del proyecto
sys.path.insert(0, str(Path(__file__).parent))
from data import DataIngestion
from pipeline import EngineeringPipeline
from sentiment import SentimentAnalyzer

# Configuración de página de Streamlit
st.set_page_config(
    page_title="Dashboard de Análisis de Sentimientos",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inicializar estados de sesión para navegación ininterrumpida
if "analizado" not in st.session_state:
    st.session_state.analizado = False
    st.session_state.raw_texts = []
    st.session_state.clean_texts = []
    st.session_state.lemmas = []
    st.session_state.results = []
    st.session_state.file_name = ""

# Inyección de estilos CSS - Tema Universal mediante translucidez y herencia
st.markdown("""
    <style>
    .metric-card {
        background-color: rgba(130, 130, 130, 0.15); /* Cristal Translúcido: se adapta al fondo nativo */
        padding: 22px;
        border-radius: 12px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.04);
        border-left: 6px solid #0066cc;
        color: inherit; /* Hereda texto blanco o negro directamente de Streamlit */
        backdrop-filter: blur(5px);
    }
    .metric-title {
        font-size: 13px;
        color: inherit;
        opacity: 0.75;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    .metric-value {
        font-size: 32px;
        font-weight: bold;
        color: inherit;
        margin-top: 6px;
    }
    </style>
""", unsafe_allow_html=True)

def configurar_graficos_universales():
    """
    Motor de renderizado agnóstico. Elimina los fondos de Matplotlib (Canal Alfa 0)
    y utiliza colores neutros para no tener que refrescar la web al cambiar de tema.
    """
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "figure.facecolor": (0, 0, 0, 0),    # Fondo de figura 100% transparente
        "axes.facecolor": (0, 0, 0, 0),      # Fondo de plano cartesiano 100% transparente
        "savefig.facecolor": (0, 0, 0, 0),   # Guardado Transparente
        "text.color": "#808495",           # Gris neutro equilibrado para claros/oscuros
        "axes.labelcolor": "#808495",
        "xtick.color": "#808495",
        "ytick.color": "#808495",
        "axes.edgecolor": "#808495",
        "grid.color": "#808495",
        "grid.alpha": 0.2
    })

class DashboardVisualizer:
    """Soporte gráfico universal con renderizado dinámico (RGBA)."""
    def __init__(self, results, lemmas):
        self.results = results
        self.lemmas = lemmas
        self.counts = Counter([r.get("label", "NEUTRO") for r in self.results])
        self.palette = {'POSITIVO': '#2ca02c', 'NEGATIVO': '#d62728', 'NEUTRO': '#1f77b4'}
        configurar_graficos_universales()

    def plot_barras(self):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        labels = list(self.counts.keys())
        values = list(self.counts.values())
        colors = [self.palette.get(l, '#7f7f7f') for l in labels]
        
        sns.barplot(x=labels, y=values, palette=colors, ax=ax)
        ax.set_title("Volumen de Opiniones por Sentimiento", fontsize=12, weight='bold', pad=12)
        ax.set_ylabel("Cantidad")
        sns.despine(left=True, bottom=True)
        fig.tight_layout()
        return fig

    def plot_pastel(self):
        fig, ax = plt.subplots(figsize=(5, 5))
        labels = list(self.counts.keys())
        values = list(self.counts.values())
        colors = [self.palette.get(l, '#7f7f7f') for l in labels]
        
        if sum(values) == 0: return fig
        
        wedges, texts, autotexts = ax.pie(
            values, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140,
            wedgeprops={'edgecolor': '#808495', 'linewidth': 0.5}
        )
        
        ax.set_title("Proporción de Sentimientos", fontsize=12, weight='bold', pad=12)
        
        # Ajuste inteligente de legibilidad interna/externa del gráfico
        for text in texts:
            text.set_color("#808495")
        for autotext in autotexts:
            autotext.set_color("white")
            autotext.set_weight("bold")
            
        fig.tight_layout()
        return fig

    def plot_wordcloud(self):
        if not self.lemmas: return None
        texto = " ".join(self.lemmas)
        
        # Generación de la Nube Semántica en Formato RGBA (Permite el paso del tema base)
        wc = WordCloud(
            width=900, height=450, 
            background_color=None, mode="RGBA", 
            colormap="plasma", max_words=100
        ).generate(texto)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        fig.tight_layout()
        return fig

    def plot_top_lemmas(self, top_n):
        if not self.lemmas: return None
        freq = Counter(self.lemmas).most_common(top_n)
        words = [item[0] for item in freq]
        counts = [item[1] for item in freq]
        
        fig, ax = plt.subplots(figsize=(8, 4.5))
        
        sns.barplot(x=counts, y=words, palette="viridis", ax=ax)
        ax.set_title(f"Top {top_n} Palabras Más Frecuentes (Lemmas)", fontsize=12, weight='bold', pad=12)
        ax.set_xlabel("Frecuencia")
        sns.despine(left=True, bottom=True)
        fig.tight_layout()
        return fig

    def plot_scores_distribution(self):
        if not self.results: return None
        scores = [r.get("score", 0.0) for r in self.results]
        
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(scores, kde=True, color="#0066cc", bins=15, ax=ax)
        ax.set_title("Distribución del Nivel de Confianza del Modelo", fontsize=12, weight='bold', pad=12)
        ax.set_xlabel("Confianza (%)")
        ax.set_ylabel("Frecuencia")
        sns.despine(left=True, bottom=True)
        fig.tight_layout()
        return fig

def main():
    st.title("🚀 Sistema de Inteligencia Artificial - Dashboard PLN")
    st.markdown("Auditoría de feedback masivo e inferencia lingüística avanzada.")
    
    # ------------------ SECCIÓN LATERIAL ------------------
    st.sidebar.header("📁 Control de Datos")
    origen_datos = st.sidebar.radio("Origen de los datos:", ["Lotes del Servidor (Samples)", "Subir mi propio archivo"])
    
    ruta_final_analisis = None
    nombre_archivo_activo = ""
    
    if origen_datos == "Lotes del Servidor (Samples)":
        data_dir = Path("data/samples")
        try:
            data_dir.mkdir(parents=True, exist_ok=True)
            archivos_validos = [f.name for f in data_dir.iterdir() if f.suffix.lower() in ['.csv', '.xlsx', '.json', '.txt']]
        except Exception:
            archivos_validos = []

        if archivos_validos:
            archivo_seleccionado = st.sidebar.selectbox("Seleccione lote del servidor:", archivos_validos)
            if archivo_seleccionado:
                ruta_final_analisis = data_dir / archivo_seleccionado
                nombre_archivo_activo = archivo_seleccionado
        else:
            st.sidebar.info("No se hallaron archivos en `data/samples/`.")
    else:
        archivo_usuario = st.sidebar.file_uploader(
            "Cargue su archivo de auditoría:", 
            type=["csv", "xlsx", "json", "txt"]
        )
        if archivo_usuario:
            dir_temporal = Path("data/uploaded_cache")
            dir_temporal.mkdir(parents=True, exist_ok=True)
            ruta_final_analisis = dir_temporal / archivo_usuario.name
            with open(ruta_final_analisis, "wb") as f:
                f.write(archivo_usuario.getbuffer())
            nombre_archivo_activo = archivo_usuario.name

    st.sidebar.markdown("---")
    ejecutar_analisis = st.sidebar.button("Procesar Lote", type="primary")
    
    # ------------------ FLUJO DEL PIPELINE ------------------
    if ejecutar_analisis and ruta_final_analisis:
        with st.spinner("Ejecutando Pipeline Lingüístico e Inferencia..."):
            try:
                ingestion = DataIngestion(ruta_final_analisis)
                textos = ingestion.load_data()
                
                pipeline = EngineeringPipeline()
                textos_limpios = []
                lemas_totales = []
                for t in textos:
                    out = pipeline.process_text(t)
                    if out and len(out.clean_text) > 1:
                        textos_limpios.append(out.clean_text)
                        lemas_totales.extend(out.lemmas)
                
                analyzer = SentimentAnalyzer()
                resultados = analyzer.analyze_batch(textos_limpios)
                
                # Salvaguardando memoria
                st.session_state.raw_texts = textos
                st.session_state.clean_texts = textos_limpios
                st.session_state.lemmas = lemas_totales
                st.session_state.results = resultados
                st.session_state.file_name = nombre_archivo_activo
                st.session_state.analizado = True
                
            except Exception as e:
                st.error(f"Falla crítica en el procesamiento: {e}")
                logging.getLogger(__name__).error("Excepción en Dashboard", exc_info=True)

    # ------------------ RENDERIZADO VISUAL ------------------
    if st.session_state.analizado:
        viz = DashboardVisualizer(st.session_state.results, st.session_state.lemmas)
        
        total = len(st.session_state.results)
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f'<div class="metric-card"><div class="metric-title">Total Reseñas</div><div class="metric-value">{total}</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><div class="metric-title">😊 Reseñas Positivas</div><div class="metric-value">{viz.counts.get("POSITIVO", 0)}</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card"><div class="metric-title">😐 Reseñas Neutras</div><div class="metric-value">{viz.counts.get("NEUTRO", 0)}</div></div>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="metric-card"><div class="metric-title">😞 Reseñas Negativas</div><div class="metric-value">{viz.counts.get("NEGATIVO", 0)}</div></div>', unsafe_allow_html=True)
            
        st.markdown(f"**Lote bajo auditoría activa:** `{st.session_state.file_name}`")
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Distribución", 
            "☁️ Nube de Palabras", 
            "📈 Palabras más Frecuentes", 
            "🎯 Confianza"
        ])
        
        # Imprimiendo gráficos con instrucción transparente a Streamlit
        with tab1:
            col_g1, col_g2 = st.columns([1.1, 0.9])
            with col_g1:
                st.pyplot(viz.plot_barras(), transparent=True)
            with col_g2:
                st.pyplot(viz.plot_pastel(), transparent=True)
                
        with tab2:
            fig_wc = viz.plot_wordcloud()
            if fig_wc:
                st.pyplot(fig_wc, transparent=True)
                
        with tab3:
            top_n = st.slider("Cantidad de palabras clave a desplegar", 5, 30, 15)
            fig_top = viz.plot_top_lemmas(top_n)
            if fig_top:
                st.pyplot(fig_top, transparent=True)
                
        with tab4:
            fig_scores = viz.plot_scores_distribution()
            if fig_scores:
                st.pyplot(fig_scores, transparent=True)
        
        st.markdown("---")
        st.subheader("📋 Detalle General de Auditoría")
        
        df_resultados = pd.DataFrame({
            'Texto Normalizado': st.session_state.clean_texts,
            'Sentimiento Detectado': [r.get('label', 'NEUTRO') for r in st.session_state.results],
            'Confianza': [f"{r.get('score', 0.0):.2f}%" for r in st.session_state.results]
        })
        
        st.dataframe(df_resultados, use_container_width=True)
        
        csv_data = df_resultados.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="💾 Descargar Resultados (CSV)",
            data=csv_data,
            file_name=f"auditoria_{st.session_state.file_name.split('.')[0]}.csv",
            mime="text/csv"
        )
    else:
        st.info("Seleccione o cargue un archivo en el panel izquierdo y presione **'Procesar Lote'**.")

if __name__ == "__main__":
    main()