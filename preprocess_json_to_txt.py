import json
import os

# Directorio de entrada con los JSON
input_dir = 'corpora/cf/json'
output_dir = 'txt/cf/'
output_file = os.path.join(output_dir, 'cfs.txt')  # Archivo combinado de salida

# Asegúrate de que el directorio de salida exista
os.makedirs(output_dir, exist_ok=True)

# Si el archivo ya existe, lo eliminamos para no duplicar datos
if os.path.exists(output_file):
    os.remove(output_file)

# Función para procesar todos los JSON válidos en un directorio
def process_all_jsons(input_dir, output_txt):
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

# Ejecutar el procesamiento
process_all_jsons(input_dir, output_file)