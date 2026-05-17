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
│   ├── distilbert_anime_pretrained/      # (Opcional) DistilBERT con MLM en dominio anime
│   ├── distilbert_bilstm_config.json     # Hiperparámetros + métricas extendidas del modelo
│   ├── distilbert_bilstm_model.pt        # Pesos entrenados del modelo final
│   ├── test_errors.csv                   # Top reseñas mal clasificadas (análisis de errores)
│   ├── test_metrics.json                 # Métricas del análisis standalone (opcional)
│   ├── confusion_matrix.png              # Heatmap de la matriz de confusión (si matplotlib)
│   └── sweeps/                           # Resultados del barrido de hiperparámetros BiLSTM
├── scripts/
│   ├── balanceo.py                       # Script para igualar clases positiva/negativa
│   ├── pretrain_mlm.py                   # Continued pretraining (MLM) sobre reseñas de anime
│   ├── entrenar_distilbert_bilstm.py     # Pipeline de entrenamiento PyTorch (Frozen/Unfrozen)
│   ├── error_analysis.py                 # Análisis de errores standalone sobre el checkpoint
│   ├── sweep_bilstm.py                   # Grid search sobre hiperparámetros del BiLSTM
│   ├── limpieza.py                       # Limpieza de texto y eliminación de duplicados
│   ├── scraper_mal.py                    # Extracción general de MyAnimeList
│   └── scraper_negativos.py              # Extracción focalizada para equilibrar clases
├── src/
│   ├── app.py                            # Interfaz web interactiva en Streamlit
│   └── model_distilbert_lstm.py          # Clase de la arquitectura neuronal (PyTorch)
├── environment.yml                       # Entorno de Conda con dependencias
├── requirements.txt                      # Alternativa pip para dependencias
└── README.md
```

## Pipeline
1. **Recolección de datos:** Extracción mediante scrapers personalizados. Se realizó un esfuerzo adicional en la recolección de reseñas negativas para mitigar el sesgo positivo inherente a los usuarios de anime (scripts/scraper_mal.py y scripts/scraper_negativos.py).
2. **Limpieza:** Procesamiento del texto para eliminar ruido y normalización de mayúsculas/minúsculas antes de la tokenización (scripts/limpieza.py).
3. **Balanceo de clases:** Generación de un dataset equilibrado (50% positivo / 50% negativo), asegurando que el modelo no aprenda sesgos estadísticos de la distribución original (scripts/balanceo.py).
4. **Continued pretraining MLM (opcional):** Adaptación del DistilBERT base al vocabulario y estilo de las reseñas de anime mediante Masked Language Modeling sobre el corpus completo sin etiquetas (scripts/pretrain_mlm.py). El modelo resultante se guarda en `models/distilbert_anime_pretrained/` y se usa como inicialización del fine-tuning supervisado.
5. **Entrenamiento en Dos Fases:** Ajuste del modelo híbrido DistilBERT + BiLSTM mediante scripts/entrenar_distilbert_bilstm.py.
    * **Fase 1 (Congelada):** Se entrenan solo la BiLSTM y el clasificador. DistilBERT actúa como un extractor de características fijo para proteger su conocimiento pre-entrenado.
    * **Fase 2 (Descongelada/Fine-tuning):** Se entrena la red completa con una tasa de aprendizaje muy baja y **layer-wise LR decay** (las capas inferiores reciben LR aún menor para preservar conocimiento general). Incluye **early stopping** sobre `val_acc`.
    * Ambas fases usan **scheduler con warmup lineal (10%)** + decay lineal y **weight decay diferenciado** (sin decay en bias y LayerNorm).
6. **Análisis de errores:** Generación de matriz de confusión, métricas extendidas (F1 por clase / macro / ponderado, precision, recall, ROC-AUC) y exportación de las reseñas mal clasificadas más "confiadas" para inspección (scripts/error_analysis.py o integrado en el entrenamiento).
7. **Inferencia/Uso:** Ejecución del modelo entrenado a través de una interfaz interactiva en Streamlit (src/app.py), con soporte opcional de **sliding window** para reseñas que excedan el `max_len` del modelo.

## Arquitectura del Modelo
* **Modelo base:** DistilBERT preentrenado (`distilbert-base-uncased` o, opcionalmente, `models/distilbert_anime_pretrained` resultado del MLM en dominio), utilizado como extractor contextual de embeddings.
* **Capa recurrente:** BiLSTM (bidireccional) que recibe como entrada el hidden size de DistilBERT. Configurable vía CLI; por defecto 256 unidades ocultas y 1 capa de profundidad.
* **Regularización:** Dropout de 0.30 aplicado antes de la capa final para prevenir el sobreajuste.
* **Capa de salida:** Capa Lineal (Dense) que reduce de 512 a 1 logit (los 512 provienen de la concatenación de las dos direcciones de la BiLSTM: 2 × 256).
* **Manejo de secuencias:** Uso de `pack_padded_sequence` en conjunto con attention masks para procesar eficientemente longitudes de texto variables.
* **Representación final:** Concatenación de los estados ocultos forward y backward de la última capa recurrente.

## Hiperparámetros y técnicas de entrenamiento

* **`max_len = 320`** (subido desde 192 para reducir el truncamiento — la reseña media ronda los 2 600 caracteres).
* **Scheduler:** `get_linear_schedule_with_warmup` con `warmup_ratio = 0.1` en cada fase.
* **Layer-wise LR decay** (F2): factor 0.9 desde el head hacia los embeddings (`--layer-lr-decay`).
* **Weight decay diferenciado:** `weight_decay = 0.01` excepto en `bias` y `LayerNorm.{weight,bias}`.
* **Early stopping** (F2): paciencia de 2 épocas sin mejora en `val_acc` (`--early-stop-patience`).
* **Mixed precision (AMP):** `torch.cuda.amp` activado en GPU.
* **Métricas guardadas en `distilbert_bilstm_config.json`:** loss, accuracy, ROC-AUC, F1/precision/recall por clase, F1 macro y ponderado, matriz de confusión.

## Sweep de hiperparámetros del BiLSTM

`scripts/sweep_bilstm.py` ejecuta un grid search sobre `lstm_hidden`, `lstm_layers` y `dropout`, guardando un checkpoint y un `config.json` por corrida bajo `models/sweeps/<run_id>/` y un `summary.csv` con los resultados ordenados por test accuracy. Ejemplo:

```
python scripts/sweep_bilstm.py --hidden 128 256 384 --layers 1 2 --dropout 0.20 0.30 0.40 \
    --epochs-frozen 3 --epochs-unfrozen 2 --max-len 256
```

## Instalación del entorno

### Conda 

conda env create -f environment.yml
conda activate anime_sentiment

### venv + pip

python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

## Ejecución

### (Opcional) Continued pretraining MLM en el dominio

```
python scripts/pretrain_mlm.py
```

Esto produce `models/distilbert_anime_pretrained/`. Luego pasarlo como base al fine-tuning supervisado:

```
python scripts/entrenar_distilbert_bilstm.py --pretrained-model models/distilbert_anime_pretrained
```

### Entrenamiento del modelo (sin pretraining previo)

```
python scripts/entrenar_distilbert_bilstm.py
```

Argumentos relevantes (todos con valor por defecto):

* `--max-len 320` — longitud de tokenización.
* `--epochs-frozen 5 --epochs-unfrozen 8` — épocas por fase.
* `--lr-head 1e-3 --lr-full 2e-5` — learning rates por fase.
* `--layer-lr-decay 0.9` — decay por capa en F2.
* `--early-stop-patience 2` — paciencia del early stopping en F2.
* `--warmup-ratio 0.1` — proporción de warmup del scheduler.
* `--pretrained-model <ruta>` — usar el DistilBERT con MLM si se entrenó.

### Análisis de errores standalone

```
python scripts/error_analysis.py
```

Reproduce el split de test, evalúa el checkpoint y genera `models/test_metrics.json`, `models/test_errors.csv` y, si `matplotlib` está instalado, `models/confusion_matrix.png`.

### Sweep del BiLSTM

```
python scripts/sweep_bilstm.py --hidden 128 256 384 --layers 1 2 --dropout 0.20 0.30 0.40
```

### Ejecutar la aplicación Streamlit

```
streamlit run src/app.py
```

La app incluye una opción de **sliding window** en la barra lateral, útil cuando la reseña excede el `max_len` del modelo.

### Hardware: Entrenamiento acelerado por hardware mediante CUDA en una GPU NVIDIA RTX 4060 / 16 GB RAM / Ryzen 5700X.