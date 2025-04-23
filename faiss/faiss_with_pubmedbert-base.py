import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import faiss
from sentence_transformers import SentenceTransformer
from metrics.metrics import load_data_semantic, precision_recall, map_vec, p_at_n, MAP, rprec, avg_prec_rec, f_beta, f1

# Cargar modelo PubMedBERT para embeddings
model = SentenceTransformer("NeuML/pubmedbert-base-embeddings")

# Leer artículos desde txt/cfs.txt
def cargar_articulos(path):
    with open(path, "r", encoding="utf-8") as f:
        contenido = f.read()
    articulos_raw = contenido.strip().split("-" * 80)
    articulos = [a.strip() for a in articulos_raw if a.strip()]
    return articulos

# Leer queries desde txt/queries.txt
def cargar_queries(path):
    with open(path, "r", encoding="utf-8") as f:
        contenido = f.read()
    queries_raw = contenido.strip().split("-" * 80)
    queries = [q.strip().replace("QueryText: ", "") for q in queries_raw if q.strip()]
    return queries

# Cargar datos
articulos = cargar_articulos("txt/cfs.txt")
queries = cargar_queries("txt/queries.txt")

# Embeddings
print("Generando embeddings de artículos...")
embeddings_cfs = model.encode(articulos, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
print("Generando embeddings de queries...")
embeddings_queries = model.encode(queries, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)

# Crear índice FAISS
index = faiss.IndexFlatIP(embeddings_cfs.shape[1])
index.add(embeddings_cfs)

# Almacenamos resultados JSON aquí
json_results = []

# Abrir el archivo de resultados para escribir
with open("results/faiss_pubmedbert.txt", "w", encoding="utf-8") as result_file:
    for i, (query_emb, query_text) in enumerate(zip(embeddings_queries, queries), 1):
        query_emb = query_emb.reshape(1, -1)
        D, I = index.search(query_emb, len(articulos))  # todos los artículos
        similitudes = D[0]
        indices = I[0]

        # Escribir top 10
        result_file.write(f"\n🔍 Query {i}:\n")
        result_file.write(f"{query_text}\n")
        result_file.write("\nTop 10 artículos más similares:\n")

        for rank, (idx, score) in enumerate(zip(indices[:10], similitudes[:10]), 1):
            result_file.write(f"{rank}. Índice: {idx + 1}, Similitud: {score:.4f}\n")
            result_file.write(f"{articulos[idx]}\n")
            result_file.write("-" * 80 + "\n")

        # Guardar todos ≥ 0.6 en JSON
        relevant_docs_json = [
            {
                "relevantDoc": int(idx + 1),  # índice original (empezando desde 1)
                "relevance": f"{score:.4f}"
            }
            for idx, score in zip(indices, similitudes) if score >= 0.5
        ]

        json_results.append({
            "queryID": i,
            "relevantDocs": relevant_docs_json
        })

# Guardar archivo JSON
with open("results/faiss_pubmedbert.json", "w", encoding="utf-8") as json_file:
    json.dump(json_results, json_file, indent=2, ensure_ascii=False)

print("Búsqueda completada. Resultados guardados en resultsCfs.txt y resultsCfs.json.")


ref_docs, semantic_docs = load_data_semantic("results/faiss_pubmedbert.json")

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