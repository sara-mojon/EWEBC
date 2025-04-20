# EWEBC: Semantic Search with FAISS and BERT Models
Participants: Sara Mojon, Adrian Bernardo, Endry Hernández, David Sueiro


This project implements a semantic search system using FAISS and pre-trained BERT models (`bioclinicalBERT` and `PubMedBERT`). It processes articles and queries, generates embeddings, and evaluates the search results using various metrics.

## Features

- **Semantic Search**: Uses FAISS for efficient similarity search.
- **Pre-trained Models**: Supports `bioclinicalBERT` and `PubMedBERT` for embedding generation.
- **Metrics**: Calculates precision, recall, F1-score, MAP, and other evaluation metrics.
- **Output**: Saves results in both `.txt` and `.json` formats.

## Requirements

- Python 3.11
- Required Python libraries:
  - `faiss`
  - `numpy`
  - `scikit-learn`
  - `sentence-transformers`
  - `tqdm`

Install dependencies using:

```bash
pip install -r requirements.txt
```

## File Structure

- `faiss_with_bioclinicalBERT.py`: Script for semantic search using `bioclinicalBERT`.
- `faiss_with_pubmedbert-base.py`: Script for semantic search using `PubMedBERT`.
- `txt/cfs.txt`: Input file containing articles.
- `txt/queries.txt`: Input file containing queries.
- `resultsCfs.txt`: Output file with top search results.
- `resultsCfs.json`: Output file with detailed results in JSON format.

## Usage

1. **Prepare Input Files**:
   - Place articles in `txt/cfs.txt`.
   - Place queries in `txt/queries.txt`.

2. **Run the Script**:
   - For `bioclinicalBERT`:
     ```bash
     python faiss_with_bioclinicalBERT.py
     ```
   - For `PubMedBERT`:
     ```bash
     python faiss_with_pubmedbert-base.py
     ```

3. **View Results**:
   - Check `resultsCfs.txt` for top 10 similar articles per query.
   - Check `resultsCfs.json` for detailed results.

## Metrics

The following metrics are calculated to evaluate the search results:

- **Precision** and **Recall**
- **F1-score** and **F2-score (F-beta)**
- **Mean Average Precision (MAP)**
- **Precision at N (P@5, P@10)**
- **R-Precision**


## Acknowledgments

- [FAISS](https://github.com/facebookresearch/faiss)
- [SentenceTransformers](https://www.sbert.net/)
- [PubMedBERT](https://huggingface.co/NeuML/pubmedbert-base-embeddings)
- [BioClinicalBERT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT)
```