# EWEBC: Semantic Search with FAISS and BERT Models
Participants: Sara Mojon, Adrian Bernardo, Endry Hernández, David Sueiro


This project implements a semantic search system using FAISS, ChromaDB, and pre-trained BERT models (`bioclinicalBERT` and `PubMedBERT`). It processes articles and queries, generates embeddings, and evaluates the search results using various metrics.
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
  - `google.generativeai`
  - `chromadb`
    
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
    - For `BERTbaseuncase`:
      ```bash
        python faiss/faiss_with_BERTbaseuncase.py
        ```
   - For `bioclinicalBERT`:
     ```bash
     python faiss/faiss_with_bioclinicalBERT.py
     ```
   - For `PubMedBERT`:
     ```bash
     python faiss/faiss_with_pubmedbert-base.py
     ```
    - For `ChromaDB`:
      ```bash
        python chromadb/chromadb_with_pubmedber-base.py
        ```

3. **View Results**:
   - Check `resultsCfs.txt` for top 10 similar articles per query.
   - Check `resultsCfs.json` for detailed results.


4. **Run RAG**:
   - For `RAG`:
     ```bash
        python rag/gemini-2.0&2.5.py  
     ```
     It will generate the `rag_gemini_responses.json` file with the generated results for the RAG model.
     ```
     python rag/qwen3_1.7B.py
     ```
     It will generate the `rag_qwen_responses.json` file with the generated results for the Qwen model.
     ```
     


## Metrics

## Metrics

# The following metrics are calculated to evaluate the search results:
# - **Precision** and **Recall**: Measure the relevance of retrieved results.
# - **F1-score** and **F2-score (F-beta)**: Harmonic mean of precision and recall.
# - **Mean Average Precision (MAP)**: Average precision across all queries.
# - **Precision at N (P@5, P@10)**: Precision for the top N results.
# - **R-Precision**: Precision at the R-th position, where R is the number of relevant documents.

## RAG METRICS

# The following metrics are calculated to evaluate the search results:

Run the following command to calculate the metrics:
```bash
python rag_metrics.py
python best_rag.py
```

The fist command will calculate the metrics for the RAG model (generate rag_XXX_metrics.json) and the second one will check the metrics for the best RAG model.


## Acknowledgments

- [FAISS](https://github.com/facebookresearch/faiss)
- [SentenceTransformers](https://www.sbert.net/)
- [PubMedBERT](https://huggingface.co/NeuML/pubmedbert-base-embeddings)
- [BioClinicalBERT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT)
- [BERTBaseUncased](https://huggingface.co/google-bert/bert-base-uncased)
```
