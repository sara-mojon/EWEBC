import json
import os
from sentence_transformers import SentenceTransformer, util
from rouge_score import rouge_scorer

# Cargar el JSON generado por OpenAI
with open("results/rag_llama32_instruct_responses.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Cargar modelo de embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # Puedes usar tu modelo original si lo prefieres
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

# Preparar almacenamiento de métricas
metricas = {}

for query_id, contenido in data.items():
    context = contenido.get("context", "")
    response = contenido.get("response", "")

    # Calcular embeddings y similaridad de coseno
    context_emb = embedding_model.encode(context, convert_to_tensor=True)
    response_emb = embedding_model.encode(response, convert_to_tensor=True)
    cosine_similarity = util.cos_sim(response_emb, context_emb).item()

    # Calcular Rouge-L F1
    rouge_score = scorer.score(context, response)
    rouge_l_f1 = rouge_score['rougeL'].fmeasure

    # Longitud de respuesta (como proxy de completitud)
    response_len = len(response.split())

    # Guardar métricas por consulta
    metricas[query_id] = {
        "cosine_similarity": round(cosine_similarity, 4),
        "rougeL_f1": round(rouge_l_f1, 4),
        "response_length_words": response_len
    }

# Guardar resultados en archivo JSON
os.makedirs("results", exist_ok=True)
with open("results/rag_llama32_instruct_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metricas, f, indent=2)

print("✅ Métricas guardadas en results/rag_qwen_metrics.json")
