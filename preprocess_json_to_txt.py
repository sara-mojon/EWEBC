import json
import os

# Directorios
input_dir = 'corpora/cf/json'
output_dir = 'txt/'
os.makedirs(output_dir, exist_ok=True)

# Diccionario de archivos de salida
output_files = {
    'cf': os.path.join(output_dir, 'cfs.txt'),
    'queries': os.path.join(output_dir, 'queries.txt')
}

# Eliminar archivos existentes para evitar duplicados
for path in output_files.values():
    if os.path.exists(path):
        os.remove(path)

# Funciones de formateo específicas por tipo
def format_cf_entry(paper):
    title = paper.get('title', 'No title available')
    abstract = paper.get('abstract/extract', 'No abstract available')
    return f"Title: {title}\nAbstract: {abstract}\n" + "-" * 80 + "\n\n"

def format_query_entry(paper):
    query = paper.get('queryText', 'No queryText available')
    return f"QueryText: {query}\n" + "-" * 80 + "\n\n"

# Mapeo de tipo a su función formateadora
formatter_map = {
    'cf': format_cf_entry,
    'queries': format_query_entry
}

# Función generalizada
def preprocess_json_files(prefix, formatter, output_path):
    count = 0
    for filename in sorted(os.listdir(input_dir)):
        if filename.endswith('.json') and filename.startswith(prefix):
            filepath = os.path.join(input_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            with open(output_path, 'a', encoding='utf-8') as out_file:
                for entry in data:
                    formatted = formatter(entry)
                    out_file.write(formatted)
                    count += 1

    print(f"✅ {count} entradas procesadas y escritas en: {output_path}")

# Ejecutar para cada tipo
preprocess_json_files('cf', formatter_map['cf'], output_files['cf'])
preprocess_json_files('queries', formatter_map['queries'], output_files['queries'])