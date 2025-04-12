import numpy as np
import faiss 
import os
from transformers import BertTokenizer, BertModel
import torch

def get_query_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

embeddings = np.load('embeddingsQueries.npy')
print(f"Se cargaron {embeddings.shape[0]} embeddings con dimensión {embeddings.shape[1]}.")
embedding_dimension = embeddings.shape[1]
num_embeddings = embeddings.shape[0]

nlist = 100  # Número de listas (clústeres). Ajustar según el tamaño del dataset.
quantizer = faiss.IndexFlatL2(embedding_dimension)  # El cuantizador también usa L2
index = faiss.IndexIVFFlat(quantizer, embedding_dimension, nlist)
index.train(embeddings)
index.add(embeddings)
print(f"Se añadieron {index.ntotal} embeddings al índice Faiss.")


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

new_query = "What are the main challenges in natural language processing?"
query_embedding = get_query_embedding(new_query)
query_embedding = np.expand_dims(query_embedding, axis=0).astype('float32') 

k = 5
# Realizar la búsqueda
distances, indices = index.search(query_embedding, k)

print(f"\nResultados de la búsqueda para la query: '{new_query}'")
for i, index_encontrado in enumerate(indices[0]):
    print(f"Vecino {i+1}: Índice {index_encontrado}, Distancia: {distances[0][i]}")

index_file = 'faiss_index.bin'
faiss.write_index(index, index_file)
print(f"\nÍndice Faiss guardado en: {index_file}")