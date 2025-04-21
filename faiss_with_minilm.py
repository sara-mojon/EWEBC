import os
import json
import torch
import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from metrics import load_data_semantic, precision_recall, norm_prec, map_vec, average_curve, p_at_n, MAP, rprec, avg_prec_rec, f_beta, f1

# === CARGAR MODELO MiniLM ===
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# === FUNCIONES DE CARGA DE TEXTOS ===
def cargar_articulos(path):
    with open(path, "r", encoding="utf-8") as f:
        contenido = f.read()
    return [a.strip() for a in contenido.strip().split("-" * 80) if a.strip()]

def cargar_queries(path):
    with open(path, "r", encoding="utf-8") as f:
        contenido = f.read()
    return [q.strip().replace("QueryText: ", "") for q in contenido.strip().split("-" * 80) if q.strip()]

# === EMBEDDING FUNCIONAL ===
def embed_text_batch(texts):
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

# === CARGA DE DATOS ===
articulos = cargar_articulos("txt/cfs.txt")
queries = cargar_queries("txt/queries.txt")

# === EMBEDDINGS ===
print("🔄 Generando embeddings de artículos con MiniLM...")
embeddings_cfs = embed_text_batch(articulos)

print("🔄 Generando embeddings de queries con MiniLM...")
embeddings_queries = embed_text_batch(queries)

# === FAISS INDEXACIÓN ===
index = faiss.IndexFlatIP(embeddings_cfs.shape[1])
index.add(embeddings_cfs)

json_results = []

# === BÚSQUEDA Y GUARDADO DE RESULTADOS ===
with open("resultsCfs.txt", "w", encoding="utf-8") as result_file:
    for i, (query_emb, query_text) in enumerate(zip(embeddings_queries, queries), 1):
        query_emb = query_emb.reshape(1, -1)
        D, I = index.search(query_emb, len(articulos))

        similitudes = D[0]
        indices = I[0]

        result_file.write(f"\n🔍 Query {i}:\n{query_text}\n\nTop 10 artículos más similares:\n")
        for rank, (idx, score) in enumerate(zip(indices[:10], similitudes[:10]), 1):
            result_file.write(f"{rank}. Índice: {idx + 1}, Similitud: {score:.4f}\n{articulos[idx]}\n")
            result_file.write("-" * 80 + "\n")

        relevant_docs_json = [
            {"relevantDoc": int(idx + 1), "relevance": f"{score:.4f}"}
            for idx, score in zip(indices, similitudes) if score >= 0.5
        ]

        json_results.append({"queryID": i, "relevantDocs": relevant_docs_json})

with open("resultsCfs.json", "w", encoding="utf-8") as json_file:
    json.dump(json_results, json_file, indent=2, ensure_ascii=False)

print("✅ Resultados guardados en resultsCfs.txt y resultsCfs.json")

# === EVALUACIÓN DE RESULTADOS ===
ref_docs, semantic_docs = load_data_semantic()

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

print("\n📊 EVALUACIÓN CON MiniLM:")
print(f"📊 Precisión media: {avg_prec_total:.4f}")
print(f"📊 Recall medio: {avg_rec_total:.4f}")
print(f"📊 F1: {f1_total:.4f}")
print(f"📊 F2 (Fbeta): {fbeta_total:.4f}")
print(f"📊 MAP: {map_score:.4f}")
print(f"📊 P@5: {p5:.4f}")
print(f"📊 P@10: {p10:.4f}")
print(f"📊 R-Precision: {rprec_total:.4f}")
