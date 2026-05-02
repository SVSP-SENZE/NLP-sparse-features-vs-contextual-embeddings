# config.py
DATASET_NAME = "ag_news"
SAMPLE_SIZE = 8000
RANDOM_STATE = 42
ENCODER_MODEL = "distilbert-base-uncased"

# Paths for caching
EMBEDDING_CACHE_PATH = "bert_embeddings.npy"
CLEANED_DATA_CACHE = "cleaned_data.pkl"