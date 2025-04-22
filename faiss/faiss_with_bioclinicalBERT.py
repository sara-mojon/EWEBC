import json
import torch
import faiss
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from metrics.metrics import load_data_semantic, precision_recall, map_vec, p_at_n, MAP, rprec, avg_prec_rec, f_beta, f1

# Cargar modelo y tokenizer
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Función para obtener el embedding de un texto (mean pooling)
def embed_text(text, tokenizer, model):
    with torch.no_grad():
        tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        tokens = {k: v.to(device) for k, v in tokens.items()}
        output = model(**tokens)
        last_hidden = output.last_hidden_state  # (batch, seq_len, hidden)
        attention_mask = tokens["attention_mask"].unsqueeze(-1).expand(last_hidden.size())
        masked_hidden = last_hidden * attention_mask
        summed = masked_hidden.sum(1)
        counts = attention_mask.sum(1)
        mean_pooled = summed / counts
        return mean_pooled.squeeze().cpu().numpy()

# Leer artículos desde txt/cfs.txt
def cargar_articulos(path):
    with open(path, "r", encoding="utf-8") as f:
        contenido = f.read()
    articulos_raw = contenido.strip().split("-" * 80)
    return [a.strip() for a in articulos_raw if a.strip()]

# Leer queries desde txt/queries.txt
def cargar_queries(path):
    with open(path, "r", encoding="utf-8") as f:
        contenido = f.read()
    queries_raw = contenido.strip().split("-" * 80)
    return [q.strip().replace("QueryText: ", "") for q in queries_raw if q.strip()]

# Cargar textos
articulos = cargar_articulos("txt/cfs.txt")
queries = cargar_queries("txt/queries.txt")

# Obtener embeddings de artículos
print("Generando embeddings de artículos...")
embeddings_cfs = []
for articulo in tqdm(articulos):
    emb = embed_text(articulo, tokenizer, model)
    embeddings_cfs.append(emb)
embeddings_cfs = np.vstack(embeddings_cfs)

# Obtener embeddings de queries
print("Generando embeddings de queries...")
embeddings_queries = []
for query in tqdm(queries):
    emb = embed_text(query, tokenizer, model)
    embeddings_queries.append(emb)
embeddings_queries = np.vstack(embeddings_queries)

# Normalizar para similitud coseno
def normalize(vectors):
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

embeddings_cfs = normalize(embeddings_cfs)
embeddings_queries = normalize(embeddings_queries)

# Crear índice FAISS
index = faiss.IndexFlatIP(embeddings_cfs.shape[1])
index.add(embeddings_cfs)

# Almacenamos resultados JSON aquí
json_results = []

# Abrir el archivo de resultados para escribir
with open("results/faiss_bioclinicalBERT.txt", "w", encoding="utf-8") as result_file:
    for i, (query_emb, query_text) in enumerate(zip(embeddings_queries, queries), 1):
        query_emb = query_emb.reshape(1, -1)
        D, I = index.search(query_emb, len(articulos))  # todos los artículos
        similitudes = D[0]
        indices = I[0]

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
                "relevantDoc": int(idx + 1),
                "relevance": f"{score:.4f}"
            }
            for idx, score in zip(indices, similitudes) if score >= 0.9
        ]

        json_results.append({
            "queryID": i,
            "relevantDocs": relevant_docs_json
        })

# Guardar archivo JSON
with open("results/faiss_bioclinicalBERT.json", "w", encoding="utf-8") as json_file:
    json.dump(json_results, json_file, indent=2, ensure_ascii=False)

print("Búsqueda completada con Bio_ClinicalBERT. Resultados en resultsCfs.txt y resultsCfs.json.")

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
