import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import google.generativeai as genai

# --------------------------------------------
# Configuración del modelo de embeddings
# --------------------------------------------
print("Cargando modelo de embeddings...")
model_embeddings = SentenceTransformer("NeuML/pubmedbert-base-embeddings")

# --------------------------------------------
# Configuración del modelo de lenguaje (Gemini)
# --------------------------------------------
print("Configurando modelo de lenguaje (Gemini)...")
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))  # Asegúrate de tener tu API key en las variables de entorno
generation_config = genai.types.GenerationConfig(
    temperature=0.3,
    top_p=0.9,
    top_k=30,
    max_output_tokens=2048,
)


gemini_flash = genai.GenerativeModel(model_name="gemini-2.0-flash", generation_config=generation_config, )
gemini_pro = genai.GenerativeModel(model_name="gemini-2.5-pro", generation_config=generation_config,)

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
#chroma_client.reset()  # Limpiar colección anterior si existía

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

with open("results/rag_gemini_responses.txt", "w", encoding="utf-8") as rag_file:
    for i, (query_emb, query_text) in enumerate(zip(embeddings_queries, queries), 1):
        print(f"\n🔍 Procesando Query {i}: {query_text}")
        rag_file.write(f"\n🔍 Query {i}:\n")
        rag_file.write(f"{query_text}\n")

        results = collection.query(
            query_embeddings=[query_emb.tolist()],
            n_results=5,  # Obtener los 5 documentos más relevantes
            include=["documents"]
        )

        relevant_documents = results["documents"][0]
        context = "\n".join(relevant_documents)

        prompt = f"Basándote en la siguiente información:\n\n{context}\n\nResponde a la siguiente pregunta: {query_text}"

        # Utilizar Gemini 2.0 Flash
        try:
            response_flash = gemini_flash.generate_content(prompt)
            rag_file.write(f"\nRespuesta de Gemini 2.0 Flash:\n{response_flash.text}\n")
            rag_results.setdefault(f"query_{i}", {}).update({"gemini_flash": response_flash.text})
        except Exception as e:
            error_message = f"Error al obtener respuesta de Gemini 2.0 Flash: {e}"
            print(error_message)
            rag_file.write(f"\n{error_message}\n")
            rag_results.setdefault(f"query_{i}", {}).update({"gemini_flash_error": error_message})

        # Utilizar Gemini 1.5 Pro
        try:
            response_pro = gemini_pro.generate_content(prompt)
            rag_file.write(f"\nRespuesta de Gemini 1.5 Pro:\n{response_pro.text}\n")
            rag_results.setdefault(f"query_{i}", {}).update({"gemini_pro": response_pro.text})
        except Exception as e:
            error_message = f"Error al obtener respuesta de Gemini 1.5 Pro: {e}"
            print(error_message)
            rag_file.write(f"\n{error_message}\n")
            rag_results.setdefault(f"query_{i}", {}).update({"gemini_pro_error": error_message})

        rag_file.write("-" * 80 + "\n")

# Guardar resultados RAG en JSON
with open("results/rag_gemini_responses.json", "w", encoding="utf-8") as json_file:
    json.dump(rag_results, json_file, indent=2, ensure_ascii=False)

print("Resultados RAG guardados en rag_gemini_responses.txt y rag_gemini_responses.json.")
print("Script completado.")