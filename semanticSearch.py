import json
import numpy as np
import os
from transformers import BertTokenizer, BertModel
import torch

# Directorio de entrada con los JSON
queries_dir = 'corpora/cf/json'
output_dir = 'txt/cf/'
output_file = os.path.join(output_dir, 'queries.txt')  # Archivo combinado de salida

# Directorio con los archivos .txt
directory = 'txt/cf'

# Asegúrate de que el directorio de salida exista
os.makedirs(output_dir, exist_ok=True)

# Si el archivo ya existe, lo eliminamos para no duplicar datos
if os.path.exists(output_file):
    os.remove(output_file)

# Cargar el modelo preentrenado y el tokenizador
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Función para procesar todos los JSON válidos en un directorio
def process_all_jsons(input_dir, output_txt):
    for filename in sorted(os.listdir(queries_dir)):
        if filename.endswith('.json') and filename.startswith('queries'):
            filepath = os.path.join(queries_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                papers = json.load(f)

            for paper in papers:
                queryText = paper.get('queryText', 'No queryText available')

                # Estructura del artículo
                query_text = f"QueryText: {queryText}\n"
                query_text += "-" * 80 + "\n\n"

                # Guardar en el archivo
                with open(output_txt, 'a', encoding='utf-8') as f_out:
                    f_out.write(query_text)

    print(f"✅ Preprocesamiento completo. Queries combinadas en: {output_txt}")

# Ejecutar el procesamiento
process_all_jsons(queries_dir, output_file)

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

