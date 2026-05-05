# SML Project: Text Classification — Sparse Features vs. Contextual Embeddings

**Course:** Statistical Machine Learning  
**Dataset:** AG News (8,000 samples, 4 classes)  
**Task:** Multi-class news topic classification

---

## Overview

This project compares classical ML classifiers trained on two types of text features:
- **Sparse features** — Bag-of-Words and TF-IDF (with dimensionality reduction via TruncatedSVD)
- **Dense embeddings** — Frozen DistilBERT [CLS] token embeddings (reduced via PCA)

No fine-tuning is performed. DistilBERT is used purely as a feature extractor.

---

## Project Structure

```
├── main.py              # Main pipeline: reduction, training, evaluation, plots
├── data_utils.py        # Data loading, preprocessing, EDA, sparse features
├── embedding_utils.py   # DistilBERT embedding extraction with caching
├── config.py            # Hyperparameters and paths
├── bert_embeddings.npy  # Cached embeddings (generated on first run)
└── cleaned_data.pkl     # Cached cleaned data (generated on first run)
```

---

## Setup

```bash
pip install torch transformers datasets scikit-learn pandas matplotlib seaborn wordcloud scipy tqdm
```

A GPU is recommended for the embedding extraction step (runs fine on CPU but is slower). The embeddings are cached after the first run so you only pay this cost once.

---

## How to Run

```bash
python main.py
```

On first run this will:
1. Download the AG News dataset from Hugging Face
2. Preprocess and run EDA (class distribution, text length plots, word clouds)
3. Extract DistilBERT embeddings and cache them to `bert_embeddings.npy`
4. Apply TruncatedSVD (BoW, TF-IDF) and PCA (BERT) — plots explained variance curves
5. Train and evaluate all 5 models on all 3 feature spaces
6. Print a full comparison table and show bar chart + heatmap

On subsequent runs, cached embeddings are loaded directly so it is much faster.

---

## Models Evaluated

| Model | Type |
|---|---|
| Logistic Regression | Supervised, linear |
| Linear SVM | Supervised, linear |
| RBF SVM | Supervised, non-linear |
| KNN (K=5) | Supervised, non-parametric |
| KMeans | Unsupervised (cluster labels mapped via Hungarian algorithm) |

---

## Metrics

- **Primary:** Macro F1-score (equal weight per class)
- **Secondary:** Accuracy, Macro Precision, Macro Recall
- **Per-class:** Full classification report printed for every model

---

## Key Findings

- BERT + Linear SVM achieves the best Macro F1 across all configurations
- Linear classifiers consistently outperform RBF SVM and KNN in both sparse and dense spaces
- TF-IDF retains ~40-55% variance at 100 SVD components; BERT retains ~85-92% at 100 PCA components
- Lower variance retention in TF-IDF does not translate to proportionally lower accuracy — the discarded components are mostly noise

---

## Configuration

Edit `config.py` to change the dataset, sample size, encoder model, or cache paths.

```python
DATASET_NAME = "ag_news"
SAMPLE_SIZE = 8000
RANDOM_STATE = 42
ENCODER_MODEL = "distilbert-base-uncased"
EMBEDDING_CACHE_PATH = "bert_embeddings.npy"
CLEANED_DATA_CACHE = "cleaned_data.pkl"
```