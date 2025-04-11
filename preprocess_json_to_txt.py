import json
import os

# Directorio de entrada con los JSON
input_dir = 'corpora/cf/json'
output_dir = 'txt/'
output_file_cfs = os.path.join(output_dir, 'cfs.txt')  # Archivo combinado de salida
output_file_queries = os.path.join(output_dir, 'queries.txt')  # Archivo de salida de queries

# Asegúrate de que el directorio de salida exista
os.makedirs(output_dir, exist_ok=True)

# Si el archivo de cfs ya existe, lo eliminamos para no duplicar datos
if os.path.exists(output_file_cfs):
    os.remove(output_file_cfs)

# Si el archivo de queries ya existe, lo eliminamos para no duplicar datos
if os.path.exists(output_file_queries):
    os.remove(output_file_queries)

# Función para procesar todos los JSON de cfs válidos en un directorio
def process_all_jsons_cfs(input_dir, output_txt):
    for filename in sorted(os.listdir(input_dir)):
        if filename.endswith('.json') and filename.startswith('cf'):
            filepath = os.path.join(input_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                papers = json.load(f)

            for paper in papers:
                title = paper.get('title', 'No title available')
                abstract = paper.get('abstract/extract', 'No abstract available')

                # Estructura del artículo
                article_text = f"Title: {title}\n"
                article_text += f"Abstract: {abstract}\n"
                article_text += "-" * 80 + "\n\n"

                # Guardar en el archivo
                with open(output_txt, 'a', encoding='utf-8') as f_out:
                    f_out.write(article_text)

    print(f"✅ Preprocesamiento completo. Artículos combinados en: {output_txt}")

# Función para procesar todos los JSON de queries válidos en un directorio
def process_all_jsons_queries(input_dir, output_txt):
    for filename in sorted(os.listdir(input_dir)):
        if filename.endswith('.json') and filename.startswith('queries'):
            filepath = os.path.join(input_dir, filename)
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

# Ejecutar el procesamiento de cfs
process_all_jsons_cfs(input_dir, output_file_cfs)

# Ejecutar el procesamiento de queries
process_all_jsons_queries(input_dir, output_file_queries)