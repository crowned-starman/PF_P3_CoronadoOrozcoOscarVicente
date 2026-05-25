"""
threads.py — EuroSAT Clasificador de Uso del Suelo
====================================================
Simula N cámaras satelitales concurrentes enviando imágenes al sistema
de predicción usando hilos (threading).

Analogía con el proyecto "Calidad de Agua":
  Agua    → medicion_sensores() devuelve [pH, turbidez, temperatura, tds]
             y los envía a predecir([ph, turbidez, temperatura, tds])

  EuroSAT → captura_satelital() selecciona una imagen aleatoria,
             extrae sus 230 columnas/variables
             y las envía a predecir([col_1, col_2, ..., col_230])

Las 230 columnas/variables que se envían a predecir() son:
  ┌──────────────┬────────┬────────────────────────────────────────────┐
  │ Grupo        │ Cols.  │ Descripción                                │
  ├──────────────┼────────┼────────────────────────────────────────────┤
  │ HOG          │  128   │ Gradientes orientados (textura/estructura) │
  │ Hist. color  │   96   │ Histograma RGB (32 bins × 3 canales)       │
  │ Stats color  │    6   │ Media y std de cada canal R, G, B          │
  ├──────────────┼────────┼────────────────────────────────────────────┤
  │ TOTAL        │  230   │                                            │
  └──────────────┴────────┴────────────────────────────────────────────┘
"""

import os
import random
import threading
import predict


# ── Configuración ──────────────────────────────────────────────────────────
N_HILOS   = 5      # Número de cámaras satelitales concurrentes por ronda
INTERVALO = 2.0    # Segundos entre rondas (configurado en main.py)

# Lock para evitar que los prints de distintos hilos se mezclen
_lock = threading.Lock()


# ── Utilidades del dataset ─────────────────────────────────────────────────

def buscar_directorio_datos(base="EuroSAT_data"):
    """
    Localiza la carpeta que contiene las subcarpetas de clase del EuroSAT.
    Es robusta ante diferentes estructuras internas del ZIP.

    Returns:
        str: Ruta a la carpeta con las 10 subcarpetas de clase.
    """
    for root, dirs, _ in os.walk(base):
        dirs_validos = [
            d for d in dirs
            if os.path.isdir(os.path.join(root, d))
            and not d.startswith(".")
            and not d.startswith("__")
        ]
        if len(dirs_validos) >= 5:
            return root
    return base


def captura_satelital(data_dir):
    """
    Simula la captura de una imagen satelital.
    Selecciona aleatoriamente una clase y una imagen dentro de ella.

    Equivalente a medicion_sensores() en el proyecto "Calidad de Agua",
    pero en lugar de generar 4 valores numéricos, selecciona una imagen
    cuyas 230 columnas serán extraídas y enviadas al modelo.

    Args:
        data_dir (str): Carpeta raíz del dataset con las subcarpetas de clase.

    Returns:
        tuple: (ruta_imagen: str, clase_real: str)
    """
    clases = [
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
        and not d.startswith(".")
    ]
    clase = random.choice(clases)
    cls_path = os.path.join(data_dir, clase)

    archivos = [
        f for f in os.listdir(cls_path)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]
    archivo = random.choice(archivos)

    return os.path.join(cls_path, archivo), clase


# ── Tarea de cada hilo ─────────────────────────────────────────────────────

def hilo_camara(thread_id, data_dir):
    """
    Tarea ejecutada por cada hilo de cámara satelital.

    Flujo:
      1. Selecciona una imagen aleatoria del dataset  (captura_satelital)
      2. Extrae sus 230 variables/columnas            (predict.extraer_features)
      3. Envía esas 230 columnas a la función predecir (predict.predecir)
      4. Imprime la etiqueta predicha y la confianza

    Args:
        thread_id (int): Identificador numérico del hilo (1, 2, 3...).
        data_dir  (str): Carpeta raíz del dataset EuroSAT.
    """
    # ── Paso 1: "capturar" imagen satelital ───────────────────────────────
    img_path, clase_real = captura_satelital(data_dir)

    # ── Paso 2: extraer las 230 columnas/variables de la imagen ──────────
    #
    #   Analogía con "Calidad de Agua":
    #     ph, turbidez, temperatura, tds = medicion_sensores()
    #
    #   Aquí en EuroSAT:
    #     col_1, col_2, ..., col_230 = extraer_features(img_path)
    #
    #   Las 230 variables son los valores numéricos que describen la imagen:
    #     128 de HOG + 96 de histograma de color + 6 de estadísticas RGB
    #
    columnas = predict.extraer_features(img_path)   # shape: (230,)

    # ── Paso 3: enviar las 230 columnas a la función de predicción ────────
    #
    #   Analogía con "Calidad de Agua":
    #     resultado = predict.predecir([ph, turbidez, temperatura, tds])
    #
    #   Aquí en EuroSAT:
    #     resultado = predict.predecir([col_1, col_2, ..., col_230])
    #
    resultado = predict.predecir(columnas)

    # ── Paso 4: mostrar resultados ────────────────────────────────────────
    acierto = "✓ CORRECTO" if resultado["etiqueta"] == clase_real else "✗ INCORRECTO"

    with _lock:   # garantiza que el bloque se imprime completo sin interferencia
        print(f"\n  [Cámara {thread_id:02d}] Imagen     : {os.path.basename(img_path)}")
        print(f"             Clase real  : {clase_real}")
        print(f"             Predicción  : {resultado['etiqueta']}")
        print(f"             Confianza   : {resultado['confianza']:.4f} "
              f"({resultado['confianza'] * 100:.1f}%)")
        print(f"             Variables   : {len(columnas)} columnas enviadas al modelo")
        print(f"             Resultado   : {acierto}")


# ── Lanzador de hilos ──────────────────────────────────────────────────────

def lanzar_hilos(data_dir, n_hilos=N_HILOS):
    """
    Crea N hilos, uno por cámara satelital, y los ejecuta en paralelo.
    Espera a que todos terminen antes de retornar.

    Cada hilo ejecuta hilo_camara() de forma independiente y concurrente.

    Args:
        data_dir (str): Carpeta raíz del dataset EuroSAT.
        n_hilos  (int): Número de cámaras/hilos a lanzar simultáneamente.
    """
    hilos = [
        threading.Thread(
            target=hilo_camara,
            args=(i, data_dir),
            name=f"Camara-{i:02d}",
        )
        for i in range(1, n_hilos + 1)
    ]

    # Lanzar todos al mismo tiempo
    for hilo in hilos:
        hilo.start()

    # Esperar a que todos terminen
    for hilo in hilos:
        hilo.join()
