import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from transformers import pipeline
import torch

torch.cuda.empty_cache()

# --------------------------------------------
# Configuración del modelo de embeddings
# --------------------------------------------
print("🔧 Cargando modelo de embeddings...")
model_embeddings = SentenceTransformer("NeuML/pubmedbert-base-embeddings")

# --------------------------------------------
# Cargar modelo de lenguaje LLaMA 3.2–1B Instruct
# --------------------------------------------
print("🤖 Cargando modelo LLaMA 3.2–1B Instruct...")
model_id = "meta-llama/Llama-3.2-1B-Instruct"

pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

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
print("🧠 Generando embeddings...")
embeddings_cfs = model_embeddings.encode(
    articulos,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=False,
    batch_size=4
)

embeddings_queries = model_embeddings.encode(
    queries,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=False,
    batch_size=4
)

del model_embeddings
torch.cuda.empty_cache()

# --------------------------------------------
# Inicializar ChromaDB
# --------------------------------------------
print("📚 Inicializando ChromaDB...")
chroma_client = chromadb.Client(Settings(
    anonymized_telemetry=False,
    persist_directory="chroma_storage"
))

collection = chroma_client.create_collection(name="cfs_collection", metadata={"hnsw:space": "cosine"})
collection.add(
    documents=articulos,
    ids=[str(i + 1) for i in range(len(articulos))],
    embeddings=embeddings_cfs.tolist(),
    metadatas=[{"original_id": i + 1} for i in range(len(articulos))]
)

# --------------------------------------------
# RAG: Buscar y generar respuestas
# --------------------------------------------
print("🔍 Ejecutando RAG con LLaMA 3.2–1B Instruct...")
rag_results = {}

with open("results/rag_llama32_instruct_responses.txt", "w", encoding="utf-8") as rag_file:
    for i, (query_emb, query_text) in enumerate(zip(embeddings_queries, queries), 1):
        print(f"\n🔍 Query {i}: {query_text}")
        rag_file.write(f"\n🔍 Query {i}:\n{query_text}\n")

        try:
            results = collection.query(
                query_embeddings=[query_emb.tolist()],
                n_results=5,
                include=["documents"]
            )
        except Exception as e:
            error_message = f"❌ Error al consultar ChromaDB: {e}"
            print(error_message)
            rag_file.write(f"\n{error_message}\n" + "-" * 80 + "\n")
            rag_results[f"query_{i}"] = {"chroma_error": error_message}
            continue

        relevant_documents = results["documents"][0]
        context = "\n".join(relevant_documents)

        # Mensaje al modelo
        messages = [
            {"role": "system", "content": "You are a helpful assistant for biomedical literature."},
            {"role": "user", "content": f"Based on the following information:\n\n{context}\n\nAnswer the following question:\n{query_text}"}
        ]

        try:
            outputs = pipe(
                messages,
                max_new_tokens=2048,
                temperature=0.7,
                top_p=0.8,
                top_k=20
            )
            output_text = outputs[0]["generated_text"][-1] if isinstance(outputs[0]["generated_text"], list) else outputs[0]["generated_text"]
            content = output_text.get("content")
            print(content)

            rag_file.write(f"\n🤖 Respuesta de LLaMA 3.2–1B Instruct:\n{content}\n")
            rag_results[f"query_{i}"] = {
                "query": query_text,
                "context": context,
                "response": content,
                "thinking": "",
                "model": model_id
            }

        except Exception as e:
            error_message = f"❌ Error al generar respuesta con LLaMA Instruct: {e}"
            print(error_message)
            rag_file.write(f"\n{error_message}\n")
            rag_results[f"query_{i}"] = {"llama_error": error_message}

        rag_file.write("-" * 80 + "\n")

# --------------------------------------------
# Guardar resultados
# --------------------------------------------
with open("results/rag_llama32_instruct_responses.json", "w", encoding="utf-8") as json_file:
    json.dump(rag_results, json_file, indent=2, ensure_ascii=False)

print("✅ Respuestas guardadas en results/rag_llama32_instruct_responses.txt y .json")
