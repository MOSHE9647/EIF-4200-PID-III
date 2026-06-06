# Sistema Inteligente de Análisis de Feedback y Sentimiento del Cliente

Un proyecto en Python para el análisis automático de reseñas y feedback en español, desarrollado como entrega del **III Proyecto de Investigación Dirigida (PID)** para la asignatura de Inteligencia Artificial — Universidad Nacional de Costa Rica, Campus Sarapiquí.

Este repositorio contiene un pipeline de procesamiento de lenguaje natural que ingiere lotes de opiniones, normaliza y lematiza los textos, y clasifica su polaridad (Positivo / Neutro / Negativo). Además incluye un dashboard para visualización de métricas y soporte para exportar reportes.

## Índice

- [Estructura del Repositorio](#-estructura-del-repositorio)
- [Características Principales](#-características-principales)
- [Requisitos de Instalación](#-requisitos-de-instalación)
- [Ejecución y Pruebas](#-ejecución-y-pruebas)
- [Ingesta de Datos](#-ingesta-de-datos)

## Estructura del Repositorio

```text
EIF-4200-PID-III/
├── data/                  # Datos de entrada (.csv, .xlsx, .json, .txt)
│   └── samples/           # Archivos de ejemplo
├── src/                   # Código fuente
│   ├── __init__.py
│   ├── data.py            # Carga e ingestión de archivos
│   ├── pipeline.py        # Limpieza, tokenización, lematización
│   ├── sentiment.py       # Clasificación de polaridad
│   └── dashboard.py       # Visualización (Streamlit / Matplotlib)
├── tests/                 # Pruebas unitarias (pytest)
├── main.py                # Orquestador / CLI
├── requirements.txt       # Dependencias
├── pytest.ini
└── README.md
```

## Características Principales

- **Pipeline Lingüístico Avanzado:** Implementa limpieza profunda, tokenización, remoción de *stop words* y lematización morfológica en español mediante la librería `spaCy` (`es_core_news_lg`).
- **Inferencia con LLMs (Transformers):** Sustituye los diccionarios estadísticos por el modelo **BETO** de Hugging Face, una arquitectura BERT optimizada para captar sarcasmo, negaciones y matices en español.
- **Renderizado "Tema Agnóstico":** Las métricas y gráficos estadísticos (Matplotlib y Seaborn) han sido programados utilizando el canal Alfa (Transparencia RGBA) e inyección de variables CSS nativas. Esto permite que **toda la aplicación se adapte automáticamente al tema Claro u Oscuro** del sistema del usuario sin necesidad de recargar la interfaz.
- **Ingesta Dinámica:** Soporta tanto el análisis de lotes desde el servidor (`data/samples/`) como la carga en caliente de archivos externos (`.csv`, `.xlsx`, `.json`, `.txt`) directamente desde la interfaz gráfica.
- **Manejo Robusto de Excepciones:** Tolerancia a fallos para archivos corruptos, textos vacíos o ruido no lingüístico, previniendo interrupciones en la ejecución del servidor.
- **Auditoría Exportable:** Capacidad de generar un Dataframe consolidado con los textos procesados, su sentimiento detectado y nivel de confianza matemática, exportable instantáneamente a un reporte CSV.

## Requisitos de Instalación (Entorno Conda)

Siga estos pasos estructurados paso a paso para desplegar y configurar el entorno virtual robusto del proyecto:

### 1. Clonar el repositorio
```bash
git clone https://github.com/MOSHE9647/EIF-4200-PID-III.git
cd EIF-4200-PID-III
```

### 2. Crear y activar el entorno virtual con Conda

Se utiliza **Python 3.11** para garantizar compatibilidad nativa estable y optimizada con las herramientas científicas.

Abrir la terminal `Anaconda Prompt` y ejecutar:
```bash
conda create --name PIDIII python=3.11 -y
conda activate PIDIII
```

### 3. Configurar el entorno de desarrollo en VSCode

Abra Visual Studio Code, instale la extensión de Python si no lo ha hecho, y configure el intérprete de Python para que apunte al entorno virtual `PIDIII` recién creado. Esto asegurará que todas las dependencias se gestionen correctamente dentro del entorno aislado.

Para configurar el intérprete, seleccione el entorno `PIDIII` desde la barra de estado inferior o usando `Ctrl+Shift+P` → `Python: Select Interpreter` → `PIDIII`.

### 4. Instalar las dependencias del proyecto

Con el entorno virtual activo, abra la terminal integrada de VSCode y ejecute el siguiente comando para instalar el archivo de requerimientos base:

```bash
pip install -r requirements.txt
```

### 5. Descargar el modelo de lenguaje de alta precisión de spaCy

Es obligatorio descargar el modelo en español de tamaño grande para optimizar el enriquecimiento sintáctico y la extracción léxica:

```bash
python -m spacy download es_core_news_lg
```

> **Nota**: En caso de que la computadora tenga limitaciones de memoria, se recomienda descargar el modelo de tamaño mediano `es_core_news_md` o el de tamaño pequeño `es_core_news_sm` y ajustar el código para cargar ese modelo en lugar del grande.

## Ejecución y Pruebas

Para arrancar el sistema, simplemente corra el siguiente comando en la terminal con el entorno virtual activo:

```bash
python main.py dashboard
```

### Otros comandos disponibles

> **Nota:** *Los comandos expuestos en el orquestador están implementados en [main.py](main.py#L1).*

```bash
# Ingesta de datos desde un archivo específico
python main.py ingest --file data/samples/comentarios_clientes.csv

# Procesamiento de datos con opción de mostrar resultados intermedios
python main.py process --file data/samples/comentarios_clientes.csv

# Procesamiento de datos de un archivo con visualización de textos procesados por el pipeline
python main.py process --file data/samples/resenas_productos.xlsx --show-processed

# Procesamiento de datos desde archivos con diferentes formatos
python main.py process --file data/samples/opiniones_empresas.json
python main.py process --file data/samples/resenas_clientes.txt

# Validación y información del sistema
python main.py validate
python main.py info

# Mostrar ayuda y comandos disponibles
python main.py --help
```

### Ejecución de pruebas unitarias

Con el entorno activo, ejecutar:

```bash
pytest -q
```

## Ingesta de Datos

El sistema incluye un módulo robusto de ingesta de datos que procesa lotes de comentarios desde múltiples fuentes.

### Formatos Soportados

- **CSV (.csv)**: Archivos de valores separados por comas
- **Excel (.xlsx)**: Archivos de hojas de cálculo
- **JSON (.json)**: Archivos de notación de objetos JavaScript
- **TXT (.txt)**: Archivos de texto plano

### Archivos de ejemplo en `data/samples/`

| Archivo | Registros | Formato |
|---------|------------------|---------|
| comentarios_clientes.csv | 15 | CSV |
| resenas_productos.xlsx   | 8  | Excel |
| opiniones_empresas.json  | 5  | JSON |
| resenas_clientes.txt     | 17 | TXT |

### Flujo de procesamiento

1. Ingesta del archivo (módulo `data.py`).
2. Normalización y limpieza (módulo `pipeline.py`).
3. Clasificación de sentimiento (módulo `sentiment.py`).
4. Visualización y exportación de resultados (`dashboard.py`, `main.py`).