import json
import os
import glob

# Ruta a los archivos de métricas
metric_folder = "results"
metric_files = glob.glob(os.path.join(metric_folder, "rag_*_metrics.json"))

# Métricas a evaluar
metric_names = ["cosine_similarity", "rougeL_f1", "response_length_words"]

# Diccionario para acumular promedios
model_scores = {}

for filepath in metric_files:
    model_name = os.path.basename(filepath).replace("rag_", "").replace("_metrics.json", "")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    scores = {metric: [] for metric in metric_names}

    for query_metrics in data.values():
        for metric in metric_names:
            if metric in query_metrics:
                scores[metric].append(query_metrics[metric])

    # Calcular promedio por métrica
    model_scores[model_name] = {metric: sum(vals) / len(vals) if vals else 0.0 for metric, vals in scores.items()}

# Mostrar resultados
print("Promedio de métricas por modelo:\n")
for model, metrics in model_scores.items():
    print(f"Modelo: {model}")
    for metric, avg in metrics.items():
        print(f"  {metric}: {avg:.4f}")
    print()

# Mejor modelo por métrica
print("🏆 Mejores modelos por métrica:\n")
for metric in metric_names:
    best_model = max(model_scores.items(), key=lambda x: x[1][metric])
    print(f"{metric}: {best_model[0]} ({best_model[1][metric]:.4f})")
