import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

torch.cuda.empty_cache()
# --------------------------------------------
# Configuración del modelo de embeddings
# --------------------------------------------
print("Cargando modelo de embeddings...")
model_embeddings = SentenceTransformer("NeuML/pubmedbert-base-embeddings")

# --------------------------------------------
# Cargar modelo de lenguaje Qwen/Qwen3-1.7B
# --------------------------------------------
print("Cargando modelo de lenguaje Qwen/Qwen3-1.7B...")
model_name = "Qwen/Qwen3-1.7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
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
print("Generando embeddings...")
embeddings_cfs = model_embeddings.encode(
    articulos,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=False,
    batch_size=4  # puedes probar con 2 o incluso 1 si sigue fallando
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
print("Inicializando base de datos vectorial (Chroma)...")
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
# Realizar búsquedas con RAG y guardar resultados
# --------------------------------------------
print("Realizando búsquedas con RAG y generando respuestas...")
rag_results = {}

with open("results/rag_qwen_responses.txt", "w", encoding="utf-8") as rag_file:
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
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False  # Thinking desactivado
        )

        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        try:
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=2048,
                temperature=0.7,
                top_p=0.8,
                top_k=20
            )
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
            output_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

            thinking_content = ""  # No se genera
            content = output_text

            rag_file.write(f"\nRespuesta de Qwen3-1.7B:\n{content}\n")
            rag_results[f"query_{i}"] = {
                "query": query_text,
                "context": context,
                "response": content,
                "thinking": thinking_content,
                "model": model_name
            }

        except Exception as e:
            error_message = f"Error al generar respuesta con Qwen3-4B: {e}"
            print(error_message)
            rag_file.write(f"\n{error_message}\n")
            rag_results[f"query_{i}"] = {"qwen3-4B_error": error_message}

        rag_file.write("-" * 80 + "\n")

# Guardar resultados RAG en JSON
with open("results/rag_qwen_responses.json", "w", encoding="utf-8") as json_file:
    json.dump(rag_results, json_file, indent=2, ensure_ascii=False)

print("Resultados RAG guardados en rag_qwen_responses.txt y rag_qwen_responses.json.")
print("Script completado.")
