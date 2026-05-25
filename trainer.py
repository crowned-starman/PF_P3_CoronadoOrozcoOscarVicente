"""
trainer.py — EuroSAT Clasificador de Uso del Suelo
====================================================
Responsabilidades:
  1. Extraer el dataset desde EuroSAT.zip
  2. Cargar imágenes y extraer el vector de 230 características por imagen
  3. Construir y entrenar la red neural con TensorFlow/Keras
  4. Guardar el modelo (.keras) y los metadatos de normalización (.npz)

Características extraídas por imagen (las "columnas" del modelo):
  - HOG  : 128 valores  (gradientes orientados — textura y estructura)
  - Color: 96  valores  (histograma RGB, 32 bins × 3 canales)
  - Stats: 6   valores  (media y std por canal R, G, B)
  ─────────────────────────────────────────────
  TOTAL : 230 columnas/variables de entrada
"""

import os
import zipfile
import random
import numpy as np
import tensorflow as tf
from PIL import Image
from skimage.feature import hog


# ── Rutas y parámetros globales ────────────────────────────────────────────
MODEL_PATH        = "modelo_eurosat.keras"
META_PATH         = "modelo_eurosat_meta.npz"
ZIP_PATH          = "EuroSAT.zip"
DATA_DIR          = "EuroSAT_data"
IMG_SIZE          = (64, 64)
SAMPLES_PER_CLASS = 300   # reducir si hay poca RAM; None = usar todo el dataset
SEED              = 42

# Parámetros de extracción HOG (deben ser idénticos en predict.py)
HOG_PARAMS = dict(
    orientations    = 8,
    pixels_per_cell = (16, 16),
    cells_per_block = (1, 1),
    channel_axis    = -1,       # imagen en formato (H, W, C)
)
N_COLOR_BINS = 32   # bins por canal para el histograma de color


# ── Funciones de extracción ────────────────────────────────────────────────

def extraer_dataset(zip_path=ZIP_PATH, data_dir=DATA_DIR):
    """
    Descomprime EuroSAT.zip si aún no se ha hecho.
    Retorna la ruta a la carpeta que contiene las subcarpetas de clase.
    """
    if not os.path.exists(data_dir):
        print(f"Extrayendo {zip_path} → {data_dir} ...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(data_dir)
        print("Extracción completada.\n")

    # Navegar hasta la carpeta que contiene las clases (puede estar anidada)
    for root, dirs, _ in os.walk(data_dir):
        dirs_validos = [
            d for d in dirs
            if os.path.isdir(os.path.join(root, d))
            and not d.startswith(".")
            and not d.startswith("__")
        ]
        if len(dirs_validos) >= 5:
            return root
    return data_dir


def extraer_features(img_path):
    """
    Convierte una imagen en su vector de 230 características (columnas).

    Columnas 1–128   : HOG — captura bordes, texturas y gradientes
    Columnas 129–224 : Histograma de color por canal (R, G, B)
    Columnas 225–230 : Media y desviación estándar de cada canal

    Args:
        img_path (str): Ruta a la imagen.

    Returns:
        np.ndarray: Vector de shape (230,) con dtype float32.
    """
    img = np.array(
        Image.open(img_path).convert("RGB").resize(IMG_SIZE),
        dtype=np.float32,
    ) / 255.0  # normalizar píxeles a [0, 1]

    # 1. HOG (128 valores)
    hog_feat = hog(img, **HOG_PARAMS).astype(np.float32)

    # 2. Histograma de color (96 valores: 32 bins × 3 canales)
    color_hist = np.concatenate([
        np.histogram(img[:, :, c], bins=N_COLOR_BINS, range=(0, 1))[0]
        for c in range(3)
    ]).astype(np.float32)

    # 3. Estadísticas de color (6 valores: media+std de R, G, B)
    color_stats = np.array([
        img[:, :, 0].mean(), img[:, :, 1].mean(), img[:, :, 2].mean(),
        img[:, :, 0].std(),  img[:, :, 1].std(),  img[:, :, 2].std(),
    ], dtype=np.float32)

    return np.concatenate([hog_feat, color_hist, color_stats])


# ── Carga del dataset ──────────────────────────────────────────────────────

def cargar_datos(data_dir, samples_per_class=SAMPLES_PER_CLASS):
    """
    Itera sobre las carpetas de clase, extrae features de cada imagen
    y construye las matrices X (features) e y (etiquetas numéricas).

    Returns:
        X           : np.ndarray (N, 230)
        y           : np.ndarray (N,)  — índices enteros
        class_names : np.ndarray — nombres de clase en orden
    """
    random.seed(SEED)

    class_names = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
        and not d.startswith(".")
    ])

    X_list, y_list = [], []

    for idx, cls in enumerate(class_names):
        cls_path = os.path.join(data_dir, cls)
        archivos = [
            f for f in os.listdir(cls_path)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ]
        if samples_per_class:
            archivos = random.sample(archivos, min(samples_per_class, len(archivos)))

        print(f"  [{idx + 1:2d}/{len(class_names)}] {cls:<25}: {len(archivos)} imágenes")

        for fname in archivos:
            try:
                feat = extraer_features(os.path.join(cls_path, fname))
                X_list.append(feat)
                y_list.append(idx)
            except Exception as e:
                print(f"    ⚠ Error en {fname}: {e}")

    return (
        np.array(X_list, dtype=np.float32),
        np.array(y_list, dtype=np.int32),
        np.array(class_names),
    )


# ── Construcción del modelo ────────────────────────────────────────────────

def construir_modelo(n_features, n_clases):
    """
    Red neural densa (fully-connected) para clasificación multiclase.

    Arquitectura:
        Entrada (230)  →  Dense(256)  →  Dropout  →
        Dense(128)  →  Dropout  →  Dense(64)  →  Dense(n_clases, softmax)
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_features,)),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.30),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.20),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(n_clases, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ── Entrenamiento ──────────────────────────────────────────────────────────

def entrenar_modelo(X, y, epochs=80, batch_size=32):
    """
    Normaliza los datos, construye el modelo y lo entrena.

    La normalización (media y std) se calcula SOLO sobre el conjunto de
    entrenamiento y se guarda en los metadatos para usarse en inferencia.

    Returns:
        model  : modelo entrenado
        mean   : np.ndarray (230,) — media por columna (train)
        std    : np.ndarray (230,) — desviación estándar por columna (train)
    """
    # Mezclar aleatoriamente
    rng = np.random.default_rng(seed=SEED)
    indices = np.arange(len(X))
    rng.shuffle(indices)
    X, y = X[indices], y[indices]

    # División 80% train / 20% validación
    split = max(1, int(0.20 * len(X)))
    X_val,   y_val   = X[:split],  y[:split]
    X_train, y_train = X[split:],  y[split:]

    # Normalización estándar (media=0, std=1)
    mean = X_train.mean(axis=0)
    std  = X_train.std(axis=0) + 1e-8   # evitar división por cero

    X_train_n = (X_train - mean) / std
    X_val_n   = (X_val   - mean) / std

    model = construir_modelo(X_train.shape[1], len(np.unique(y)))
    model.summary()

    print(f"\nEntrenando — {len(X_train)} muestras de train, {len(X_val)} de validación\n")
    model.fit(
        X_train_n, y_train,
        validation_data=(X_val_n, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
    )

    val_loss, val_acc = model.evaluate(X_val_n, y_val, verbose=0)
    print(f"\nValidación final — loss: {val_loss:.4f} | accuracy: {val_acc:.4f}")

    return model, mean, std


# ── Guardado ───────────────────────────────────────────────────────────────

def guardar_modelo(model, mean, std, class_names,
                   model_path=MODEL_PATH, meta_path=META_PATH):
    """
    Persiste el modelo en formato .keras y los metadatos de normalización
    junto con los nombres de clase en un archivo .npz.
    """
    model.save(model_path)
    np.savez(meta_path, mean=mean, std=std, class_names=class_names)
    print(f"\n✔ Modelo guardado en    : {model_path}")
    print(f"✔ Metadatos guardados en: {meta_path}")


# ── Pipeline completo ──────────────────────────────────────────────────────

def entrenar_y_guardar(zip_path=ZIP_PATH, model_path=MODEL_PATH, meta_path=META_PATH):
    """Ejecuta el pipeline completo: extraer → cargar → entrenar → guardar."""
    data_dir = extraer_dataset(zip_path)
    print(f"Dataset encontrado en: {data_dir}\n")
    print("Extrayendo características de las imágenes...\n")

    X, y, class_names = cargar_datos(data_dir)

    print(f"\n{'─'*55}")
    print(f"  Muestras totales : {len(X):,}")
    print(f"  Clases           : {len(class_names)}")
    print(f"  Columnas/Features: {X.shape[1]}")
    print(f"{'─'*55}\n")

    model, mean, std = entrenar_modelo(X, y)
    guardar_modelo(model, mean, std, class_names, model_path, meta_path)


# ── Ejecución directa ──────────────────────────────────────────────────────
if __name__ == "__main__":
    entrenar_y_guardar()
