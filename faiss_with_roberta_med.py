import os
import json
import torch
import faiss
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from metrics import load_data_semantic, precision_recall, norm_prec, map_vec, average_curve, p_at_n, MAP, rprec, avg_prec_rec, f_beta, f1

# === CARGAR MODELO RoBERTa-Med ===
model_name = "allenai/biomed_roberta_base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# === FUNCIÓN DE EMBEDDING ===
def embed_text(text, tokenizer, model):
    with torch.no_grad():
        tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        tokens = {k: v.to(device) for k, v in tokens.items()}
        output = model(**tokens)
        hidden = output.last_hidden_state
        attention_mask = tokens["attention_mask"].unsqueeze(-1).expand(hidden.size())
        summed = (hidden * attention_mask).sum(1)
        counts = attention_mask.sum(1)
        return (summed / counts).squeeze().cpu().numpy()

# === FUNCIÓN DE CARGA ===
def cargar_articulos(path):
    with open(path, "r", encoding="utf-8") as f:
        return [a.strip() for a in f.read().strip().split("-" * 80) if a.strip()]

def cargar_queries(path):
    with open(path, "r", encoding="utf-8") as f:
        return [q.strip().replace("QueryText: ", "") for q in f.read().strip().split("-" * 80) if q.strip()]

articulos = cargar_articulos("txt/cfs.txt")
queries = cargar_queries("txt/queries.txt")

# === EMBEDDINGS ===
print("🔄 Generando embeddings de artículos con SciFive...")
embeddings_cfs = np.vstack([embed_text(a, tokenizer, model) for a in tqdm(articulos)])

print("🔄 Generando embeddings de queries con SciFive...")
embeddings_queries = np.vstack([embed_text(q, tokenizer, model) for q in tqdm(queries)])

# === NORMALIZACIÓN + INDEXACIÓN ===
def normalize(vectors):
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

embeddings_cfs = normalize(embeddings_cfs)
embeddings_queries = normalize(embeddings_queries)
index = faiss.IndexFlatIP(embeddings_cfs.shape[1])
index.add(embeddings_cfs)

# === BÚSQUEDA + GUARDADO ===
json_results = []
with open("resultsCfs.txt", "w", encoding="utf-8") as result_file:
    for i, (query_emb, query_text) in enumerate(zip(embeddings_queries, queries), 1):
        query_emb = query_emb.reshape(1, -1)
        D, I = index.search(query_emb, len(articulos))
        similitudes, indices = D[0], I[0]

        result_file.write(f"\n🔍 Query {i}:\n{query_text}\n\nTop 10 artículos más similares:\n")
        for rank, (idx, score) in enumerate(zip(indices[:10], similitudes[:10]), 1):
            result_file.write(f"{rank}. Índice: {idx + 1}, Similitud: {score:.4f}\n{articulos[idx]}\n{'-' * 80}\n")

        json_results.append({
            "queryID": i,
            "relevantDocs": [{"relevantDoc": int(idx + 1), "relevance": f"{score:.4f}"} for idx, score in zip(indices, similitudes) if score >= 0.5]
        })

with open("resultsCfs.json", "w", encoding="utf-8") as json_file:
    json.dump(json_results, json_file, indent=2, ensure_ascii=False)

# === EVALUACIÓN ===
ref_docs, semantic_docs = load_data_semantic()
map_vals, p5, p10, rprec_total, f1_total, fbeta_total = [], 0, 0, 0, 0, 0
avg_prec_total, avg_rec_total = 0, 0

for i in range(len(ref_docs)):
    rel = ref_docs[i]["relevantDocs"]
    sem = semantic_docs[i]["relevant_docs"]
    map_vals.append(map_vec(rel, sem))
    p5 += p_at_n(rel, sem, 5)
    p10 += p_at_n(rel, sem, 10)
    avg_prec, avg_rec = avg_prec_rec(rel, sem)
    avg_prec_total += avg_prec
    avg_rec_total += avg_rec
    f1_total += f1(avg_prec, avg_rec)
    fbeta_total += f_beta(avg_prec, avg_rec, 2)
    rprec_total += rprec(rel, sem)

n = len(ref_docs)
print("\n📊 EVALUACIÓN CON RoBERTa_biomed:")
print(f"📊 Precisión media: {avg_prec_total / n:.4f}")
print(f"📊 Recall medio: {avg_rec_total / n:.4f}")
print(f"📊 F1: {f1_total / n:.4f}")
print(f"📊 F2 (Fbeta): {fbeta_total / n:.4f}")
print(f"📊 MAP: {MAP(map_vals, ref_docs):.4f}")
print(f"📊 P@5: {p5 / n:.4f}")
print(f"📊 P@10: {p10 / n:.4f}")
print(f"📊 R-Precision: {rprec_total / n:.4f}")
