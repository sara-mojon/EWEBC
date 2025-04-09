import json
import os

# Ruta al archivo JSON específico
input_file = 'corpora/cf/json/cf75.json'  # Ruta a tu archivo JSON
output_dir = 'txt/cf'  # Directorio donde se guardará el archivo txt

# Asegúrate de que el directorio de salida exista
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Nombre del archivo de salida
output_file = os.path.join(output_dir, 'cf75.txt')

# Si el archivo ya existe, lo eliminamos para no duplicar datos
if os.path.exists(output_file):
    os.remove(output_file)

# Función para procesar el archivo JSON
def process_json_file(json_file, output_txt):
    # Cargar el archivo JSON
    with open(json_file, 'r', encoding='utf-8') as f:
        papers = json.load(f)

    # Recorrer cada artículo
    for paper in papers:
        title = paper.get('title', 'No title available')
        authors = ', '.join(paper.get('authors', ['No authors available']))
        abstract = paper.get('abstract/extract', 'No abstract available')

        # Crear un texto con la información estructurada
        article_text = f"Title: {title}\n"
        article_text += f"Authors: {authors}\n"
        article_text += f"Abstract: {abstract}\n"
        article_text += "-" * 80 + "\n\n"  # Separador visual entre artículos

        # Guardar la información en el archivo (modo append)
        with open(output_txt, 'a', encoding='utf-8') as f_out:
            f_out.write(article_text)

# Procesar el archivo JSON
process_json_file(input_file, output_file)

print("Preprocesamiento completo. El archivo .txt ha sido generado.")
