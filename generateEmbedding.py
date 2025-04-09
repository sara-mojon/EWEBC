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
article_count = 0  # Contador de artículos

for filename in os.listdir(directory):
    if filename.endswith('.txt'):
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            text = file.read()
            articulos = text.split('--------------------------------------------------------------------------------')
            for articulo in articulos:
                articulo = articulo.strip()
                if articulo:  # Evitar vacíos
                    embedding = get_embedding(articulo)
                    embeddings.append(embedding)
                    article_count += 1
                    print(f"Artículo {article_count} procesado.")  # Imprime cada vez que se genera un embedding

# Guardar los embeddings en un archivo .npy
np.save('embeddings.npy', np.array(embeddings))

print(f"Se generaron {article_count} embeddings.")

