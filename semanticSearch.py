import json
import numpy as np
import os
from transformers import BertTokenizer, BertModel
import torch

# Directorio con los archivos .txt
directory = 'txt/cf'

# Cargar el modelo preentrenado y el tokenizador
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Función para obtener el embedding de un texto
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    # Tomamos la representación del token [CLS]
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

# Procesar todos los archivos .txt
embeddings = []
queries_count = 0  # Contador de queries

for filename in os.listdir(directory):
    if filename.endswith('.txt') and filename.startswith('queries'):
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            text = file.read()
            queries = text.split('--------------------------------------------------------------------------------')
            for query in queries:
                query = query.strip()
                if query:  # Evitar vacíos
                    embedding = get_embedding(query)
                    print(f"Shape del embedding de la query {queries_count + 1}: {embedding.shape}")  # Añadimos un print para ver la longitud del embedding
                    embeddings.append(embedding)
                    queries_count += 1
                    print(f"Query {queries_count} procesada.")

# Guardar los embeddings en un archivo .npy
np.save('embeddingsQueries.npy', np.array(embeddings))

print(f"Se generaron {queries_count} embeddings.")

