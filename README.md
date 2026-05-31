# Sistema Inteligente de Análisis de Feedback y Sentimiento del Cliente

Este sistema modular de Procesamiento de Lenguaje Natural (PLN) ha sido desarrollado en Python como cumplimiento del **IV Proyecto de Investigación Dirigida (PID)** para el curso de Inteligencia Artificial I de la Universidad Nacional de Costa Rica, Campus Sarapiquí.

El software realiza ingesta por lotes de opiniones de comercio electrónico, aplica normalización avanzada (limpieza, tokenización, remoción de *stop words*, lematización) y clasifica la polaridad del sentimiento, desplegando un dashboard estadístico visual enfocado en la toma de decisiones gerenciales.

## Aporte: Pipeline de Ingeniería

La sección de pipeline implementa la limpieza, tokenización avanzada, remoción de *stop words* y lematización de opiniones en español mediante spaCy. Esta capa prepara los textos normalizados para que puedan ser usados posteriormente por el análisis de sentimiento y las visualizaciones del sistema.

## Estructura del Repositorio

```text
EIF-4200-PID-III/
│
├── data/                  # Carpeta para colocar los archivos .txt, .csv o .xlsx
│   └── samples/           # Datos de prueba de reseñas de clientes
│
├── src/                   # Código fuente del sistema inteligente
│   ├── data.py            # Clase para ingesta de datos (CSV, Excel, TXT)
│   ├── pipeline.py        # Clase para limpieza, tokenización y lematización
│   ├── sentiment.py       # Clase para análisis de polaridad (Positivo/Neutro/Negativo)
│   └── dashboard.py       # Clase para generación de gráficos y WordCloud
│
├── main.py                # Punto de entrada de la aplicación (Consola/Orquestador)
├── requirements.txt       # Dependencias del proyecto
├── .gitignore             # Archivos omitidos en Git
└── README.md              # Documentación de instalación y uso
```

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

## Ejecución del Sistema

Para arrancar el sistema, simplemente corra el siguiente comando en la terminal con el entorno virtual activo:

```bash
python main.py
```
