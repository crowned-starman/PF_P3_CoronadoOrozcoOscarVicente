"""
main.py — EuroSAT Clasificador de Uso del Suelo
=================================================
Punto de entrada del sistema.

Flujo de ejecución:
  1. Intenta cargar el modelo entrenado (modelo_eurosat.keras)
  2. Si no existe → ejecuta el entrenamiento completo (trainer.py)
  3. Localiza el directorio del dataset EuroSAT_data/
  4. Lanza N hilos en bucle continuo (threads.py)
     Cada hilo = una cámara satelital que envía 230 columnas a predecir()

Uso:
    python main.py
"""

import time
import predict
import trainer
import threads

# ── Rutas del modelo ───────────────────────────────────────────────────────
MODEL_PATH = "modelo_eurosat.keras"
META_PATH  = "modelo_eurosat_meta.npz"


def banner():
    print("\n" + "=" * 57)
    print("   CLASIFICADOR DE USO DEL SUELO — EuroSAT")
    print("   Red Neural Densa + Extracción de Características")
    print("=" * 57)
    print("   Clases : AnnualCrop, Forest, HerbaceousVegetation,")
    print("            Highway, Industrial, Pasture,")
    print("            PermanentCrop, Residential, River, SeaLake")
    print("   Columnas/Variables por imagen: 230")
    print("     • HOG           : 128 (textura y estructura)")
    print("     • Histograma RGB:  96 (distribución de color)")
    print("     • Stats RGB     :   6 (media y std por canal)")
    print("=" * 57 + "\n")


def main():
    banner()

    # ── Fase 1: Verificar si existe el modelo ──────────────────────────────
    cargado = predict.cargar_modelo_y_meta(MODEL_PATH, META_PATH)

    if cargado is None:
        print("No se encontró modelo entrenado.")
        print("Iniciando entrenamiento completo...\n")
        trainer.entrenar_y_guardar(
            model_path=MODEL_PATH,
            meta_path=META_PATH,
        )
        print("\nEntrenamiento finalizado.")
    else:
        _, _, _, class_names = cargado
        print(f"Modelo cargado. {len(class_names)} clases disponibles.")

    # ── Fase 2: Localizar el dataset ───────────────────────────────────────
    data_dir = threads.buscar_directorio_datos()
    print(f"Dataset localizado en: {data_dir}")

    # ── Fase 3: Bucle de predicción por hilos ─────────────────────────────
    print(f"\nIniciando sistema de predicción con {threads.N_HILOS} hilos concurrentes.")
    print("Presiona Ctrl+C para detener.\n")

    ronda = 1
    try:
        while True:
            print(f"\n{'─' * 57}")
            print(f"  Ronda {ronda:>3} — {threads.N_HILOS} cámaras satelitales en paralelo")
            print(f"{'─' * 57}")
            threads.lanzar_hilos(data_dir, n_hilos=threads.N_HILOS)
            ronda += 1
            time.sleep(threads.INTERVALO)

    except KeyboardInterrupt:
        print(f"\n\n{'─' * 57}")
        print("  Sistema detenido por el usuario.")
        print(f"  Rondas completadas: {ronda - 1}")
        print(f"{'─' * 57}\n")


if __name__ == "__main__":
    main()
