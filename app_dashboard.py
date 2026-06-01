"""
Dashboard Interactivo de Análisis de Sentimientos
Interfaz Streamlit para visualizar y analizar sentimientos en tiempo real
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
from collections import Counter

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data import DataIngestion
from pipeline import EngineeringPipeline
from sentiment import SentimentAnalyzer
from dashboard import SentimentMetrics, DashboardVisualizer

# Configurar página
st.set_page_config(
    page_title="Dashboard de Análisis de Sentimientos",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-title {
        font-size: 14px;
        color: #666;
        font-weight: 600;
    }
    .metric-value {
        font-size: 32px;
        color: #0066cc;
        font-weight: bold;
    }
    .header-title {
        text-align: center;
        color: #0066cc;
        margin-bottom: 30px;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_components():
    """Carga los componentes del sistema (cachado para rendimiento)."""
    analyzer = SentimentAnalyzer()
    pipeline = EngineeringPipeline()
    return analyzer, pipeline

@st.cache_data
def load_data_from_samples():
    """Carga todos los datos de data/samples/."""
    data_dir = Path("data/samples")
    all_texts = []
    files_loaded = []
    
    if not data_dir.exists():
        return [], []
    
    supported_ext = {'.csv', '.xlsx', '.json', '.txt'}
    files = [f for f in data_dir.iterdir() if f.suffix.lower() in supported_ext]
    
    for file_path in files:
        try:
            ingestion = DataIngestion(file_path)
            texts = ingestion.load_data()
            all_texts.extend(texts)
            files_loaded.append(f"{file_path.name} ({len(texts)} textos)")
        except Exception as e:
            st.warning(f"Error cargando {file_path.name}: {e}")
            continue
    
    return all_texts, files_loaded

def create_sentiment_chart(results):
    """Crea gráfico interactivo de distribución de sentimientos."""
    metrics = SentimentMetrics(results)
    distribution = metrics.get_distribution()
    
    # Gráfico de barras
    fig = go.Figure()
    
    colors = {
        'POSITIVO': '#00CC96',
        'NEUTRO': '#636EFA',
        'NEGATIVO': '#EF553B'
    }
    
    fig.add_trace(go.Bar(
        x=list(distribution.keys()),
        y=list(distribution.values()),
        marker=dict(
            color=[colors.get(label, '#636EFA') for label in distribution.keys()],
            line=dict(color='black', width=2)
        ),
        text=[f"{v:.1f}%" for v in distribution.values()],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Porcentaje: %{y:.1f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title="Distribución de Sentimientos",
        xaxis_title="Sentimiento",
        yaxis_title="Porcentaje (%)",
        height=400,
        showlegend=False,
        hovermode='x unified'
    )
    
    return fig

def create_pie_chart(results):
    """Crea gráfico de pastel."""
    metrics = SentimentMetrics(results)
    distribution = metrics.get_distribution()
    
    colors = {
        'POSITIVO': '#00CC96',
        'NEUTRO': '#636EFA',
        'NEGATIVO': '#EF553B'
    }
    
    fig = go.Figure(data=[go.Pie(
        labels=list(distribution.keys()),
        values=list(distribution.values()),
        marker=dict(
            colors=[colors.get(label, '#636EFA') for label in distribution.keys()],
            line=dict(color='white', width=2)
        ),
        textposition='inside',
        textinfo='label+percent',
        hovertemplate='<b>%{label}</b><br>Porcentaje: %{value:.1f}%<extra></extra>'
    )])
    
    fig.update_layout(
        title="Distribución de Sentimientos (Pastel)",
        height=400,
        showlegend=True
    )
    
    return fig

def create_wordcloud_from_lemmas(lemmas):
    """Crea WordCloud con plotly."""
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    import io
    from PIL import Image
    
    if not lemmas:
        return None
    
    text = " ".join(lemmas)
    
    # Generar WordCloud
    wordcloud = WordCloud(
        width=1200,
        height=600,
        background_color="white",
        colormap="viridis",
        max_words=100
    ).generate(text)
    
    # Convertir a imagen
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    plt.tight_layout(pad=0)
    
    # Guardar en buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return Image.open(buf)

def create_top_lemmas_chart(lemmas, top_n=15):
    """Crea gráfico de palabras más frecuentes."""
    if not lemmas:
        return None
    
    lemma_counts = Counter(lemmas)
    top_lemmas = lemma_counts.most_common(top_n)
    
    if not top_lemmas:
        return None
    
    words, frequencies = zip(*top_lemmas)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=list(words),
        x=list(frequencies),
        orientation='h',
        marker=dict(
            color=list(frequencies),
            colorscale='Viridis',
            line=dict(color='black', width=1)
        ),
        text=list(frequencies),
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Frecuencia: %{x}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f"Top {top_n} Palabras Más Frecuentes",
        xaxis_title="Frecuencia",
        yaxis_title="Palabra",
        height=500,
        showlegend=False,
        hovermode='y unified'
    )
    
    return fig

def create_scores_chart(results):
    """Crea gráfico de scores promedio."""
    metrics = SentimentMetrics(results)
    avg_scores = metrics.get_average_scores()
    
    colors = {
        'POSITIVO': '#00CC96',
        'NEUTRO': '#636EFA',
        'NEGATIVO': '#EF553B'
    }
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=list(avg_scores.keys()),
        y=list(avg_scores.values()),
        marker=dict(
            color=[colors.get(label, '#636EFA') for label in avg_scores.keys()],
            line=dict(color='black', width=2)
        ),
        text=[f"{v:.1f}" for v in avg_scores.values()],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Score Promedio: %{y:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Score Promedio por Sentimiento",
        xaxis_title="Sentimiento",
        yaxis_title="Score Promedio",
        height=400,
        showlegend=False,
        yaxis=dict(range=[0, 100]),
        hovermode='x unified'
    )
    
    return fig

def main():
    """Función principal de la aplicación."""
    
    # Header
    st.markdown("<h1 class='header-title'> Dashboard de Análisis de Sentimientos</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    st.sidebar.markdown("##  Configuración")
    
    with st.spinner(" Cargando componentes..."):
        analyzer, pipeline = load_components()
    
    st.sidebar.success(" Componentes cargados")
    
    # Opción de carga de datos
    data_source = st.sidebar.radio(
        "Selecciona la fuente de datos:",
        [" Cargar desde samples/", " Cargar archivo personalizado", "🔨 Usar datos de ejemplo"]
    )
    
    all_texts = []
    files_loaded = []
    
    if data_source == " Cargar desde samples/":
        with st.spinner(" Cargando datos de samples/..."):
            all_texts, files_loaded = load_data_from_samples()
        
        if files_loaded:
            st.sidebar.success(f" Se cargaron {len(files_loaded)} archivo(s)")
            with st.sidebar.expander(" Archivos cargados"):
                for file in files_loaded:
                    st.text(f"• {file}")
    
    elif data_source == " Cargar archivo personalizado":
        uploaded_file = st.sidebar.file_uploader(
            "Sube un archivo (CSV, Excel, JSON, TXT)",
            type=['csv', 'xlsx', 'json', 'txt']
        )
        
        if uploaded_file:
            # Guardar temporal
            temp_path = Path("temp_upload") / uploaded_file.name
            temp_path.parent.mkdir(exist_ok=True)
            temp_path.write_bytes(uploaded_file.getbuffer())
            
            try:
                ingestion = DataIngestion(temp_path)
                all_texts = ingestion.load_data()
                st.sidebar.success(f"Se cargaron {len(all_texts)} registros")
            except Exception as e:
                st.sidebar.error(f" Error: {e}")
    
    else:  # Datos de ejemplo
        all_texts = [
            "Excelente producto, superó mis expectativas.",
            "Muy bueno, lo recomiendo.",
            "Perfecto, exactamente lo que necesitaba.",
            "Terrible calidad, no funciona.",
            "Muy caro para lo que ofrece.",
            "Producto defectuoso, decepción total.",
            "Normal, nada especial.",
            "Funciona bien pero podría mejorar.",
            "Servicio impecable, muy satisfecho.",
            "Horrible experiencia de compra.",
        ]
        st.sidebar.info(" Usando datos de ejemplo")
    
    if not all_texts:
        st.warning(" No hay datos para analizar. Por favor carga datos.")
        return
    
    # Mostrar resumen de datos
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Resumen de Datos")
    st.sidebar.metric("Total de textos", len(all_texts))
    
    # Procesar datos
    if st.sidebar.button(" Procesar y Analizar", key="process_btn"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Paso 1: Pipeline
            status_text.text("1/3 Procesando textos con NLP...")
            progress_bar.progress(33)
            processed_texts = pipeline.process_batch(all_texts)
            all_lemmas = []
            for processed in processed_texts:
                all_lemmas.extend(processed['lemmas'])
            
            # Paso 2: Sentimiento
            status_text.text("2/3 Analizando sentimientos...")
            progress_bar.progress(66)
            sentiment_results = analyzer.analyze_batch(all_texts)
            
            # Paso 3: Visualizaciones
            status_text.text("3/3 Generando visualizaciones...")
            progress_bar.progress(100)
            
            st.session_state.processed = True
            st.session_state.results = sentiment_results
            st.session_state.lemmas = all_lemmas
            st.session_state.metrics = SentimentMetrics(sentiment_results)
            
            status_text.empty()
            progress_bar.empty()
            st.success(" ¡Análisis completado!")
            
        except Exception as e:
            st.error(f" Error en el análisis: {e}")
    
    # Mostrar resultados si están disponibles
    if 'results' in st.session_state and st.session_state.processed:
        st.markdown("---")
        st.markdown("##  Resultados del Análisis")
        
        # Métricas principales
        metrics = st.session_state.metrics
        summary = metrics.get_summary()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(" Total de Textos", summary["total_reviews"])
        
        with col2:
            dist = summary["distribution"]
            positivos = dist.get('POSITIVO', 0)
            st.metric(" Sentimiento Positivo", f"{positivos:.1f}%")
        
        with col3:
            negativos = dist.get('NEGATIVO', 0)
            st.metric(" Sentimiento Negativo", f"{negativos:.1f}%")
        
        with col4:
            neutros = dist.get('NEUTRO', 0)
            st.metric(" Sentimiento Neutro", f"{neutros:.1f}%")
        
        st.markdown("---")
        
        # Gráficos
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["📊 Barras", " Pastel", " Nube de Palabras", " Top Palabras", " Scores"]
        )
        
        with tab1:
            st.plotly_chart(create_sentiment_chart(st.session_state.results), use_container_width=True)
        
        with tab2:
            st.plotly_chart(create_pie_chart(st.session_state.results), use_container_width=True)
        
        with tab3:
            st.markdown("### Nube de Palabras Dinámicas")
            wordcloud_img = create_wordcloud_from_lemmas(st.session_state.lemmas)
            if wordcloud_img:
                st.image(wordcloud_img, use_column_width=True)
            else:
                st.warning("No se pudo generar la nube de palabras")
        
        with tab4:
            top_n = st.slider("Selecciona cantidad de palabras", 5, 30, 20)
            fig = create_top_lemmas_chart(st.session_state.lemmas, top_n)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        with tab5:
            st.plotly_chart(create_scores_chart(st.session_state.results), use_container_width=True)
        
        st.markdown("---")
        
        # Tabla de resultados detallados
        st.markdown("##  Resultados Detallados")
        
        results_df = pd.DataFrame({
            'Texto': all_texts,
            'Sentimiento': [r['label'] for r in st.session_state.results],
            'Score': [r['score'] for r in st.session_state.results]
        })
        
        st.dataframe(results_df, use_container_width=True)
        
        # Descargar resultados
        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=" Descargar resultados (CSV)",
            data=csv,
            file_name="analisis_sentimientos.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    # Inicializar session state
    if 'processed' not in st.session_state:
        st.session_state.processed = False
    
    main()
