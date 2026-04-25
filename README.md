# Análisis de Sentimiento de Reseñas de Anime

**Integrantes:** Diego Silva Madariaga / Bastián Cortez Arce

**Profesor:** Jaime Andrés Jiménez / Procesamiento de lenguaje natural

**Institución:** UTEM (Universidad Tecnológica Metropolitana)

**Periodo:** Primer Semestre 2026


## Descripción del Proyecto
Este proyecto utiliza una arquitectura híbrida de Deep Learning (DistilBERT + BiLSTM) para clasificar reseñas de MyAnimeList como positivas o negativas.

## Estructura del Directorio

```text
ANALISIS DE SENTIMIENTO ANIME LIST/
├── data/
│   ├── raw/                              # Datasets originales obtenidos por scraping
│   │   ├── anime_sentiment.csv
│   │   ├── resenas_backup.csv
│   │   ├── resenas_negativos_backup.csv
│   │   └── resenas_negativos.csv
│   ├── anime_sentiment_balanceado.csv    # Dataset final listo para entrenamiento (50/50)
│   ├── anime_sentiment_limpio.csv        # Dataset procesado general
│   └── resenas_negativos_limpio.csv      # Dataset procesado clase minoritaria
├── models/
│   ├── distilbert_tokenizer/             # Archivos del AutoTokenizer guardados
│   ├── distilbert_bilstm_config.json     # Hiperparámetros del modelo
│   └── distilbert_bilstm_model.pt        # Pesos entrenados del modelo final
├── scripts/
│   ├── balanceo.py                       # Script para igualar clases positiva/negativa
│   ├── entrenar_distilbert_bilstm.py     # Pipeline de entrenamiento PyTorch (Frozen/Unfrozen)
│   ├── limpieza.py                       # Limpieza de texto y eliminación de duplicados
│   ├── scraper_mal.py                    # Extracción general de MyAnimeList
│   └── scraper_negativos.py              # Extracción focalizada para equilibrar clases
├── src/
│   ├── app.py                            # Interfaz web interactiva en Streamlit
│   └── model_distilbert_lstm.py          # Clase de la arquitectura neuronal (PyTorch)
├── tests/                                # Scripts de pruebas unitarias
├── environment.yml                       # Entorno de Conda con dependencias
├── requirements.txt                      # Alternativa pip para dependencias
└── README.md
```

## Pipeline
1. **Recolección de datos:** Extracción mediante scrapers personalizados. Se realizó un esfuerzo adicional en la recolección de reseñas negativas para mitigar el sesgo positivo inherente a los usuarios de anime.(scripts/scraper_mal.py y scripts/scraper_negativos.py)
2. **Limpieza:** Procesamiento del texto para eliminar ruido y normalización de mayúsculas/minúsculas antes de la tokenización (scripts/limpieza.py).
3. **Balanceo de clases:** Generación de un dataset equilibrado (50% positivo / 50% negativo), asegurando que el modelo no aprenda sesgos estadísticos de la distribución original (scripts/balanceo.py).
4. **Entrenamiento en Dos Fases:** Ajuste del modelo híbrido DistilBERT + BiLSTM mediante scripts/entrenar_distilbert_bilstm.py.
    * Fase 1 (Congelada): Se entrenan solo la BiLSTM y el clasificador. DistilBERT actúa como un extractor de características fijo para proteger su conocimiento pre-entrenado.
    * Fase 2 (Descongelada/Fine-tuning): Se entrena la red completa con una tasa de aprendizaje muy baja, permitiendo que DistilBERT se adapte al vocabulario técnico del anime.
5. **Inferencia/Uso:** Ejecución del modelo entrenado a través de una interfaz interactiva en Streamlit (src/app.py).

## Arquitectura del Modelo
* **Modelo base:** DistilBERT preentrenado (distilbert-base-uncased), utilizado como extractor contextual de embeddings.
* **Capa recurrente:** BiLSTM (bidireccional) que recibe como entrada el hidden size de DistilBERT. Configurada con un tamaño oculto de 256 y 1 capa de profundidad.
* **Regularización:** Dropout de 0.30 aplicado antes de la capa final para prevenir el sobreajuste.
* **Capa de salida:** Capa Lineal (Dense) que reduce de 512 a 1 logit (Los 512 provienen de la concatenación de las dos direcciones de la BiLSTM: 2 x 256).
* **Manejo de secuencias:** Uso de pack_padded_sequence en conjunto con attention masks para procesar eficientemente longitudes de texto variables.
* **Representación final:** Concatenación de los estados ocultos forward y backward de la última capa recurrente.

## Instalación del entorno

### Conda 

conda env create -f environment.yml
conda activate anime_sentiment

### venv + pip

python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

## Ejecución

### Entrenamiento del modelo

cd scripts
python entrenar_distilbert_bilstm.py

### Ejecutar la aplicación Streamlit

streamlit run src/app.py

### Hardware: Entrenamiento acelerado por hardware mediante CUDA en una GPU NVIDIA RTX 4060/16 GB RAM/Ryzen 5700x.