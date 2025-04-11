import numpy as np
import os
from transformers import BertTokenizer, BertModel
import torch

# Directorio con los archivos .txt
directory = 'txt'

# Cargar el modelo preentrenado y el tokenizador
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Función genérica para obtener el embedding de un texto
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

# Función para procesar un archivo y generar embeddings
def generate_embeddings(input_file, output_npy, label='Elemento'):
    embeddings = []
    count = 0

    with open(os.path.join(directory, input_file), 'r', encoding='utf-8') as file:
        text = file.read()
        chunks = text.split('--------------------------------------------------------------------------------')
        for chunk in chunks:
            chunk = chunk.strip()
            if chunk:
                embedding = get_embedding(chunk)
                embeddings.append(embedding)
                count += 1
                print(f"{label} {count} procesado.")

    np.save(output_npy, np.array(embeddings))
    print(f"✅ Se generaron {count} embeddings en {output_npy}.\n")

# Generar embeddings para cfs.txt
generate_embeddings('cfs.txt', 'embeddingsCfs.npy', label='Artículo')

# Generar embeddings para queries.txt
generate_embeddings('queries.txt', 'embeddingsQueries.npy', label='Query')
