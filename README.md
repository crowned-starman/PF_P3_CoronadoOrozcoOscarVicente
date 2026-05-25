#  EuroSAT — Clasificador de Uso del Suelo

Sistema de clasificación de imágenes satelitales con **Red Neural Densa + TensorFlow/Keras**,
basado en extracción de características y predicción concurrente por hilos.

---

##  ¿De qué se trata?

El proyecto clasifica imágenes de satélite del dataset **EuroSAT** en 10 categorías de uso
del suelo europeo:

| # | Clase | Descripción |
|---|-------|-------------|
| 1 | `AnnualCrop`             | Cultivos anuales |
| 2 | `Forest`                 | Bosques |
| 3 | `HerbaceousVegetation`   | Vegetación herbácea |
| 4 | `Highway`                | Carreteras y autopistas |
| 5 | `Industrial`             | Zonas industriales |
| 6 | `Pasture`                | Pastizales |
| 7 | `PermanentCrop`          | Cultivos permanentes |
| 8 | `Residential`            | Zonas residenciales |
| 9 | `River`                  | Ríos |
| 10| `SeaLake`                | Mar y lagos |

Las imágenes provienen del satélite **Sentinel-2** y tienen una resolución de **64 × 64 píxeles RGB**.

---

##  ¿Cómo funciona?

El sistema sigue el mismo patrón que el proyecto **"Calidad de Agua"**, adaptado para visión satelital:

```
    EuroSAT (este proyecto)
 ─────────────────────────────────────────
    captura_satelital()
    → imagen aleatoria del dataset

       230 variables / columnas extraídas:
        128 HOG  (textura y estructura)
         96 Hist. de color (R, G, B)
          6 Stats de color (media + std)

 predecir([col_1, col_2, ..., col_230])
 → "Forest" | "River" | etc.
```

### Pipeline completo

```
EuroSAT.zip
    │
    ▼ trainer.py
┌─────────────────────────────────────────────────────────────────┐
│  1. Extrae el ZIP → EuroSAT_data/                               │
│  2. Por cada imagen extrae 230 columnas (HOG + color + stats)   │
│  3. Construye red neural: 230 → 256 → 128 → 64 → 10 (softmax)  │
│  4. Entrena y guarda:                                           │
│       modelo_eurosat.keras   (red entrenada)                    │
│       modelo_eurosat_meta.npz (media, std, nombres de clase)    │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼ main.py (al ejecutar)
┌─────────────────────────────────────────────────────────────────┐
│  1. Carga modelo (o entrena si no existe)                       │
│  2. Lanza N hilos concurrentes (threads.py)                     │
│     Cada hilo (cámara satelital):                               │
│       a. Selecciona imagen aleatoria       captura_satelital()  │
│       b. Extrae sus 230 columnas           extraer_features()   │
│       c. Envía las 230 columnas al modelo  predecir(columnas)   │
│       d. Imprime etiqueta + confianza                           │
└─────────────────────────────────────────────────────────────────┘
```

### Las 230 columnas/variables del modelo

```
Columnas   1 – 128  →  HOG (Histogram of Oriented Gradients)
                        Captura bordes, texturas y gradientes de la imagen.
                        Configuración: 8 orientaciones, celdas 16×16 px.

Columnas 129 – 224  →  Histograma de color RGB
                        Distribución de intensidades por canal.
                        32 bins × 3 canales (R, G, B) = 96 valores.

Columnas 225 – 230  →  Estadísticas de color
                        Media de R, media de G, media de B,
                        std de R,   std de G,   std de B.
```

### Arquitectura de la red neural

```
Entrada  (230)  →  Dense(256, relu)  →  Dropout(0.30)
                →  Dense(128, relu)  →  Dropout(0.20)
                →  Dense(64,  relu)
                →  Dense(10,  softmax)  →  clase predicha
```

---

##  Estructura del proyecto

```
eurosat_clasificador/
├── main.py           # Punto de entrada: carga modelo y lanza hilos
├── trainer.py        # Extracción de features, entrenamiento y guardado
├── predict.py        # Carga del modelo y función predecir()
├── threads.py        # Hilos concurrentes que envían datos a predecir()
├── EuroSAT.zip       # Dataset (debes colocarlo aquí antes de ejecutar)
├── requirements.txt  # Dependencias del proyecto
├── .gitignore        # Archivos excluidos de Git
└── README.md         # Este archivo
```

> Los archivos `modelo_eurosat.keras`, `modelo_eurosat_meta.npz` y la carpeta
> `EuroSAT_data/` se generan automáticamente al ejecutar `main.py` por primera vez.

---

##  Cómo ejecutar desde cero

### Requisitos previos

- **Python 3.11** (requerido por TensorFlow)
- **EuroSAT.zip** colocado en la carpeta raíz del proyecto

### Paso 1 — Crear entorno virtual con Python 3.11

```bash
# Windows
py -3.11 -m venv venv
venv\Scripts\activate

# macOS / Linux
python3.11 -m venv venv
source venv/bin/activate
```

### Paso 2 — Instalar dependencias

```bash
pip install -r requirements.txt
```

### Paso 3 — Colocar el dataset

Asegúrate de que `EuroSAT.zip` esté en la carpeta raíz del proyecto
(al mismo nivel que `main.py`).

```
eurosat_clasificador/
├── EuroSAT.zip   ← aquí
├── main.py
├── ...
```

### Paso 4 — Ejecutar el sistema

```bash
python main.py
```

**Primera ejecución** — El sistema detecta que no hay modelo y ejecuta
automáticamente el entrenamiento completo (~5–15 min según el hardware):

```
==========================================================
   CLASIFICADOR DE USO DEL SUELO — EuroSAT
==========================================================
No se encontró modelo entrenado.
Iniciando entrenamiento completo...

Extrayendo EuroSAT.zip → EuroSAT_data/ ...
Cargando imágenes y extrayendo características...

  [ 1/10] AnnualCrop            : 300 imágenes
  [ 2/10] Forest                : 300 imágenes
  ...

Entrenando — 2400 muestras de train, 600 de validación

Epoch 1/80 ...
...

✔ Modelo guardado en    : modelo_eurosat.keras
✔ Metadatos guardados en: modelo_eurosat_meta.npz
```

**Ejecuciones posteriores** — Carga el modelo directamente e inicia las predicciones:

```
Modelo cargado. 10 clases disponibles.

──────────────────────────────────────────────────────────
  Ronda   1 — 5 cámaras satelitales en paralelo
──────────────────────────────────────────────────────────

  [Cámara 01] Imagen     : Forest_00042.jpg
               Clase real  : Forest
               Predicción  : Forest
               Confianza   : 0.9231 (92.3%)
               Variables   : 230 columnas enviadas al modelo
               Resultado   : ✓ CORRECTO

  [Cámara 03] Imagen     : River_00187.jpg
               Clase real  : River
               Predicción  : River
               Confianza   : 0.8754 (87.5%)
               Variables   : 230 columnas enviadas al modelo
               Resultado   : ✓ CORRECTO
  ...
```

Presiona **Ctrl+C** para detener el sistema.

---

##  Parámetros configurables

| Parámetro | Archivo | Valor por defecto | Descripción |
|-----------|---------|-------------------|-------------|
| `SAMPLES_PER_CLASS` | `trainer.py` | `300` | Imágenes por clase para entrenar. `None` = todas |
| `N_HILOS`  | `threads.py` | `5` | Cámaras satelitales concurrentes |
| `INTERVALO` | `threads.py` | `2.0` | Segundos entre rondas |
| `epochs` | `trainer.py` | `80` | Épocas de entrenamiento |

---

##  Dependencias

| Librería | Uso |
|----------|-----|
| `tensorflow` | Construcción y entrenamiento de la red neural |
| `numpy` | Álgebra vectorial y matrices |
| `Pillow` | Carga y redimensionado de imágenes |
| `scikit-image` | Extracción de características HOG |

---

##  Notas

- El archivo `EuroSAT.zip` **no se sube a Git** (ver `.gitignore`).
- Los archivos del modelo (`.keras`, `.npz`) tampoco se suben; se regeneran con `python main.py`.
- Solo se sube el **código fuente** (`*.py`, `requirements.txt`, `README.md`, `.gitignore`).

---

