"""
predict.py — EuroSAT Clasificador de Uso del Suelo
====================================================
Responsabilidades:
  1. Cargar el modelo entrenado (.keras) y los metadatos (.npz)
  2. Exponer la función extraer_features() — idéntica a la de trainer.py
  3. Exponer predecir(features) — recibe los 230 valores y retorna la etiqueta

Nota: extraer_features() y los HOG_PARAMS DEBEN ser idénticos a los de
trainer.py para garantizar coherencia entre entrenamiento e inferencia.
"""

import os
import numpy as np
import tensorflow as tf
from PIL import Image
from skimage.feature import hog


# ── Rutas y parámetros (deben coincidir con trainer.py) ───────────────────
MODEL_PATH   = "modelo_eurosat.keras"
META_PATH    = "modelo_eurosat_meta.npz"
IMG_SIZE     = (64, 64)

HOG_PARAMS = dict(
    orientations    = 8,
    pixels_per_cell = (16, 16),
    cells_per_block = (1, 1),
    channel_axis    = -1,
)
N_COLOR_BINS = 32


# ── Carga de modelo y metadatos ────────────────────────────────────────────

def cargar_modelo_y_meta(model_path=MODEL_PATH, meta_path=META_PATH):
    """
    Carga el modelo Keras y los metadatos de normalización/clases.

    Returns:
        (model, mean, std, class_names) si ambos archivos existen.
        None si alguno falta (indica que hay que entrenar primero).
    """
    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        return None

    model       = tf.keras.models.load_model(model_path)
    meta        = np.load(meta_path, allow_pickle=True)
    mean        = meta["mean"].astype(np.float32)
    std         = meta["std"].astype(np.float32)
    class_names = meta["class_names"]

    return model, mean, std, class_names


# ── Extracción de características ─────────────────────────────────────────

def extraer_features(img_path):
    """
    Convierte una imagen en su vector de 230 características (columnas).

    Las columnas representan las "variables de entrada" del modelo:
      Col.   1–128 : HOG (gradientes orientados — textura/estructura)
      Col. 129–224 : Histograma RGB (32 bins × 3 canales)
      Col. 225–230 : Media y std por canal (R, G, B)

    Args:
        img_path (str): Ruta a la imagen a clasificar.

    Returns:
        np.ndarray: Vector de shape (230,) con dtype float32.
    """
    img = np.array(
        Image.open(img_path).convert("RGB").resize(IMG_SIZE),
        dtype=np.float32,
    ) / 255.0

    # 1. HOG — 128 valores
    hog_feat = hog(img, **HOG_PARAMS).astype(np.float32)

    # 2. Histograma de color — 96 valores
    color_hist = np.concatenate([
        np.histogram(img[:, :, c], bins=N_COLOR_BINS, range=(0, 1))[0]
        for c in range(3)
    ]).astype(np.float32)

    # 3. Estadísticas de color — 6 valores
    color_stats = np.array([
        img[:, :, 0].mean(), img[:, :, 1].mean(), img[:, :, 2].mean(),
        img[:, :, 0].std(),  img[:, :, 1].std(),  img[:, :, 2].std(),
    ], dtype=np.float32)

    return np.concatenate([hog_feat, color_hist, color_stats])


# ── Función principal de predicción ───────────────────────────────────────

def predecir(features, model_path=MODEL_PATH, meta_path=META_PATH):
    """
    Recibe el vector de 230 variables/columnas y retorna la predicción.

    Este es el equivalente directo a:
        predict.predecir([ph, turbidez, temperatura, tds])
    del proyecto "Calidad de Agua", pero con 230 columnas en lugar de 4.

    Args:
        features (list | np.ndarray): Vector de 230 valores float —
            las mismas 230 columnas con las que se entrenó el modelo.

    Returns:
        dict con las claves:
            "etiqueta"       : str  — clase predicha (ej. "Forest")
            "confianza"      : float — probabilidad de la clase ganadora
            "probabilidades" : dict  — probabilidad por cada clase
    """
    cargado = cargar_modelo_y_meta(model_path, meta_path)

    if cargado is None:
        raise FileNotFoundError(
            "No se encontró modelo o metadatos. "
            "Ejecuta primero el entrenamiento con: python trainer.py"
        )

    model, mean, std, class_names = cargado

    # Normalizar con los estadísticos del entrenamiento
    x      = np.asarray(features, dtype=np.float32).reshape(1, -1)
    x_norm = (x - mean) / std

    probs    = model.predict(x_norm, verbose=0)[0]
    pred_idx = int(np.argmax(probs))

    return {
        "etiqueta"       : str(class_names[pred_idx]),
        "confianza"      : float(probs[pred_idx]),
        "probabilidades" : {
            str(class_names[i]): float(probs[i])
            for i in range(len(class_names))
        },
    }


# ── Wrapper conveniente ────────────────────────────────────────────────────

def predecir_desde_imagen(img_path):
    """
    Atajo que extrae features de una imagen y llama a predecir().

    Args:
        img_path (str): Ruta a la imagen.

    Returns:
        dict — mismo formato que predecir().
    """
    features = extraer_features(img_path)
    return predecir(features)
