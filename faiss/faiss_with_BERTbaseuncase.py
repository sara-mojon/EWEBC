import json
import sys
import numpy as np
import torch
import faiss
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from metrics.metrics import load_data_semantic, precision_recall, map_vec, p_at_n, MAP, rprec, avg_prec_rec, f_beta, f1

# Crear carpeta de resultados si no existe
os.makedirs("results", exist_ok=True)

# Cargar modelo BERT base uncased
print("Cargando modelo BERT...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Función para obtener embedding (CLS token)
def embed_text(text):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        return outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

# Leer textos desde archivos
print("Leyendo textos...")
def cargar_textos(path):
    with open(path, "r", encoding="utf-8") as f:
        bloques = f.read().split('-' * 80)
    return [b.strip().replace("QueryText: ", "") for b in bloques if b.strip()]

articulos = cargar_textos("txt/cfs.txt")
queries = cargar_textos("txt/queries.txt")

# Generar embeddings de artículos
print("Generando embeddings de artículos...")
embeddings_cfs = [embed_text(a) for a in tqdm(articulos)]
embeddings_cfs = np.vstack(embeddings_cfs)

# Generar embeddings de queries
print("Generando embeddings de queries...")
embeddings_queries = [embed_text(q) for q in tqdm(queries)]
embeddings_queries = np.vstack(embeddings_queries)

# Normalizar
def normalize(vectors):
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

embeddings_cfs = normalize(embeddings_cfs)
embeddings_queries = normalize(embeddings_queries)

# Crear índice FAISS
print("Creando índice FAISS...")
index = faiss.IndexFlatIP(embeddings_cfs.shape[1])
index.add(embeddings_cfs)

# Buscar y guardar resultados
json_results = []
with open("results/faiss_bert_uncased.txt", "w", encoding="utf-8") as result_file:
    for i, (query_emb, query_text) in enumerate(zip(embeddings_queries, queries), 1):
        D, I = index.search(query_emb.reshape(1, -1), len(articulos))
        similitudes, indices = D[0], I[0]

        # Top 10 para el TXT
        result_file.write(f"\n🔍 Query {i}:\n")
        result_file.write(f"{query_text}\n")
        result_file.write("\nTop 10 artículos más similares:\n")
        for rank, (idx, score) in enumerate(zip(indices[:10], similitudes[:10]), 1):
            result_file.write(f"{rank}. Índice: {idx}, Similitud: {score:.4f}\n")
            result_file.write(f"{articulos[idx]}\n")
            result_file.write("-" * 80 + "\n")

        # Guardar todos ≥ 0.6 en el JSON
        relevant_docs_json = [
            {"relevantDoc": int(idx), "relevance": f"{score:.4f}"}
            for idx, score in zip(indices, similitudes) if score >= 0.6
        ]

        json_results.append({
            "queryID": i,
            "relevantDocs": relevant_docs_json
        })

# Guardar el JSON
with open("results/faiss_bert_uncased.json", "w", encoding="utf-8") as f_json:
    json.dump(json_results, f_json, indent=2, ensure_ascii=False)

print("\n✅ Búsqueda completada con BERT base uncased. Resultados guardados en 'results/faiss_bert_uncased.txt' y 'results/faiss_bert_uncased.json'")

ref_docs, semantic_docs = load_data_semantic("results/faiss_bioclinicalBERT.json")

map_vals, prec_vals = [], []
p5, p10, rprec_total, f1_total, fbeta_total = 0, 0, 0, 0, 0
avg_prec_total, avg_rec_total = 0, 0

for i in range(len(ref_docs)):
    rel_docs = ref_docs[i]["relevantDocs"]
    sem_docs = semantic_docs[i]["relevant_docs"]

    prec_vals.append(precision_recall(rel_docs, sem_docs))
    map_vals.append(map_vec(rel_docs, sem_docs))
    p5 += p_at_n(rel_docs, sem_docs, 5)
    p10 += p_at_n(rel_docs, sem_docs, 10)

    avg_prec, avg_rec = avg_prec_rec(rel_docs, sem_docs)
    avg_prec_total += avg_prec
    avg_rec_total += avg_rec
    f1_total += f1(avg_prec, avg_rec)
    fbeta_total += f_beta(avg_prec, avg_rec, 2)

    rprec_total += rprec(rel_docs, sem_docs)

n = len(ref_docs)
avg_prec_total /= n
avg_rec_total /= n
f1_total /= n
fbeta_total /= n
p5 /= n
p10 /= n
rprec_total /= n
map_score = MAP(map_vals, ref_docs)

print(f"📊 Precisión media: {avg_prec_total:.4f}")
print(f"📊 Recall medio: {avg_rec_total:.4f}")
print(f"📊 F1: {f1_total:.4f}")
print(f"📊 F2 (Fbeta): {fbeta_total:.4f}")
print(f"📊 MAP: {map_score:.4f}")
print(f"📊 P@5: {p5:.4f}")
print(f"📊 P@10: {p10:.4f}")
print(f"📊 R-Precision: {rprec_total:.4f}")
