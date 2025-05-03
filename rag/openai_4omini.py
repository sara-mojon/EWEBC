import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from openai import OpenAI

# --------------------------------------------
# Configuración del modelo de embeddings
# --------------------------------------------
print("Cargando modelo de embeddings...")
model_embeddings = SentenceTransformer("NeuML/pubmedbert-base-embeddings")

# --------------------------------------------
# Configuración del cliente OpenAI
# --------------------------------------------
print("Inicializando cliente OpenAI...")
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))  # Asegúrate de definir esta variable de entorno

openai_model = "gpt-4o-mini"  # Cambia a "gpt-4" si lo prefieres

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
embeddings_cfs = model_embeddings.encode(articulos, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=False)
embeddings_queries = model_embeddings.encode(queries, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=False)

# --------------------------------------------
# Inicializar ChromaDB
# --------------------------------------------
print("Inicializando base de datos vectorial (Chroma)...")
chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))

collection = chroma_client.create_collection(name="cfs_collection", metadata={"hnsw:space": "cosine"})
collection.add(
    documents=articulos,
    ids=[str(i + 1) for i in range(len(articulos))],
    embeddings=embeddings_cfs.tolist(),
    metadatas=[{"original_id": i + 1} for i in range(len(articulos))]
)

# --------------------------------------------
# Realizar búsquedas con RAG y guardar resultados
# --------------------------------------------
print("Realizando búsquedas con RAG y generando respuestas...")
rag_results = {}

with open("results/rag_openai_responses.txt", "w", encoding="utf-8") as rag_file:
    for i, (query_emb, query_text) in enumerate(zip(embeddings_queries, queries), 1):
        print(f"\n🔍 Procesando Query {i}: {query_text}")
        rag_file.write(f"\n🔍 Query {i}:\n{query_text}\n")

        results = collection.query(
            query_embeddings=[query_emb.tolist()],
            n_results=5,
            include=["documents"]
        )

        relevant_documents = results["documents"][0]
        print(f"RELEVANT_DOCUMENTS: {relevant_documents}")
        context = "\n".join(relevant_documents)

        prompt = f"Based on the following information:\n\n{context}\n\nAnswer the following question: {query_text}"

        try:
            completion = client.chat.completions.create(
                model=openai_model,
                messages=[
                    {"role": "system",
                     "content": "Eres un asistente útil que responde de forma precisa basada en contexto científico."},
                    {"role": "user", "content": prompt}
                ]
            )
            response_text = completion.choices[0].message.content.strip()

            rag_file.write(f"\nRespuesta de OpenAI ({openai_model}):\n{response_text}\n")
            rag_results[f"query_{i}"] = {
                "query": query_text,
                "context": context,
                "response": response_text,
                "model": openai_model
            }

        except Exception as e:
            error_message = f"Error al generar respuesta con OpenAI: {e}"
            print(error_message)
            rag_file.write(f"\n{error_message}\n")
            rag_results[f"query_{i}"] = {
                "query": query_text,
                "context": context,
                "error": error_message,
                "model": openai_model
            }

# Guardar resultados RAG en JSON
with open("results/rag_openai_responses.json", "w", encoding="utf-8") as json_file:
    json.dump(rag_results, json_file, indent=2, ensure_ascii=False)

print("Resultados RAG guardados en rag_openai_responses.txt y rag_openai_responses.json.")
print("Script completado.")
