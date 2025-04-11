import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Cargar los embeddings
embeddings_cfs = np.load('embeddingsCfs.npy')        # Artículos
embeddings_queries = np.load('embeddingsQueries.npy')  # Queries

# Leer los textos originales para mostrar los resultados
with open("txt/cfs.txt", "r", encoding="utf-8") as f:
    texto_cfs = f.read()
    articulos = [a.strip() for a in texto_cfs.split('-' * 80) if a.strip()]

with open("txt/queries.txt", "r", encoding="utf-8") as f:
    texto_queries = f.read()
    queries = [q.strip() for q in texto_queries.split('-' * 80) if q.strip()]

# Verificación básica
assert len(embeddings_cfs) == len(articulos), "Mismatch entre artículos y embeddings."
assert len(embeddings_queries) == len(queries), "Mismatch entre queries y embeddings."

# Abrir el archivo de resultados para escribir
with open("resultsCfs.txt", "w", encoding="utf-8") as result_file:
    # Calcular similitudes para cada query
    for i, (query_emb, query_text) in enumerate(zip(embeddings_queries, queries), 1):
        query_emb = query_emb.reshape(1, -1)
        similitudes = cosine_similarity(query_emb, embeddings_cfs)[0]
        top_indices = np.argsort(similitudes)[::-1][:10]

        # Escribir los resultados en el archivo
        result_file.write(f"\n🔍 Query {i}:\n")
        result_file.write(f"{query_text}\n")
        result_file.write("\nTop 10 artículos más similares:\n")

        for rank, idx in enumerate(top_indices, 1):
            result_file.write(f"{rank}. Índice: {idx}, Similitud: {similitudes[idx]:.4f}\n")
            result_file.write(f"{articulos[idx]}\n")
            result_file.write("-" * 80 + "\n")

print("Los resultados se han guardado en 'resultsCfs.txt'.")
