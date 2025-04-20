import os
import json
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from metrics import (
    load_data_semantic, precision_recall, map_vec, p_at_n, MAP,
    rprec, avg_prec_rec, f_beta, f1
)

# --------------------------------------------
# Configuración del modelo de embeddings
# --------------------------------------------
print("Cargando modelo...")
model = SentenceTransformer("NeuML/pubmedbert-base-embeddings")

# --------------------------------------------
# Cargar archivos de entrada
# --------------------------------------------
def cargar_articulos(path):
    with open(path, "r", encoding="utf-8") as f:
        contenido = f.read()
    articulos_raw = contenido.strip().split("-" * 80)
    return [a.strip() for a in articulos_raw if a.strip()]

def cargar_queries(path):
    with open(path, "r", encoding="utf-8") as f:
        contenido = f.read()
    queries_raw = contenido.strip().split("-" * 80)
    return [q.strip().replace("QueryText: ", "") for q in queries_raw if q.strip()]

articulos = cargar_articulos("txt/cfs.txt")
queries = cargar_queries("txt/queries.txt")

# --------------------------------------------
# Generar embeddings
# --------------------------------------------
print("Generando embeddings...")
embeddings_cfs = model.encode(articulos, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
embeddings_queries = model.encode(queries, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)

# --------------------------------------------
# Inicializar ChromaDB
# --------------------------------------------
print("Inicializando base de datos vectorial (Chroma)...")
chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))
#chroma_client.reset()  # Limpiar colección anterior si existía

collection = chroma_client.create_collection(name="cfs_collection")
collection.add(
    documents=articulos,
    ids=[str(i + 1) for i in range(len(articulos))],
    embeddings=embeddings_cfs.tolist(),
    metadatas=[{"original_id": i + 1} for i in range(len(articulos))]
)

# --------------------------------------------
# Realizar búsquedas y guardar resultados
# --------------------------------------------
json_results = []

with open("resultsCfs.txt", "w", encoding="utf-8") as result_file:
    for i, (query_emb, query_text) in enumerate(zip(embeddings_queries, queries), 1):
        result_file.write(f"\n🔍 Query {i}:\n")
        result_file.write(f"{query_text}\n")
        result_file.write("\nTop 10 artículos más similares:\n")

        results = collection.query(
            query_embeddings=[query_emb.tolist()],
            n_results=len(articulos),
            include=["documents", "distances", "metadatas"]  # eliminamos "ids"
        )

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        scores = results["distances"][0]
        original_ids = [meta["original_id"] for meta in metas]

        for rank, (doc_id, score) in enumerate(zip(original_ids[:10], scores[:10]), 1):
            idx = doc_id
            result_file.write(f"{rank}. Índice: {idx}, Similitud: {score:.4f}\n")
            result_file.write(f"{articulos[idx - 1]}\n")
            result_file.write("-" * 80 + "\n")

        # Guardar todos ≥ 0.5 en JSON
        relevant_docs_json = [
            {
                "relevantDoc": int(doc_id),
                "relevance": f"{score:.4f}"
            }
            for doc_id, score in zip(original_ids, scores) if score >= 1.9
        ]

        json_results.append({
            "queryID": i,
            "relevantDocs": relevant_docs_json
        })

# Guardar archivo JSON
with open("resultsCfs.json", "w", encoding="utf-8") as json_file:
    json.dump(json_results, json_file, indent=2, ensure_ascii=False)

print("Resultados guardados en resultsCfs.txt y resultsCfs.json.")

# --------------------------------------------
# Evaluación de métricas semánticas
# --------------------------------------------
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

print(f"\n📊 Precisión media: {avg_prec_total:.4f}")
print(f"📊 Recall medio: {avg_rec_total:.4f}")
print(f"📊 F1: {f1_total:.4f}")
print(f"📊 F2 (Fbeta): {fbeta_total:.4f}")
print(f"📊 MAP: {map_score:.4f}")
print(f"📊 P@5: {p5:.4f}")
print(f"📊 P@10: {p10:.4f}")
print(f"📊 R-Precision: {rprec_total:.4f}")
