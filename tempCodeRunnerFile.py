import torch
import numpy as np
import os
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import config

def get_bert_embeddings(text_list):
    """
    Extracts [CLS] embeddings from a frozen pretrained encoder.

    """
    
    # 1. Check if cached embeddings already exist to save time
    if os.path.exists(config.EMBEDDING_CACHE_PATH):
        print(f"--- Loading cached embeddings from {config.EMBEDDING_CACHE_PATH} ---")
        return np.load(config.EMBEDDING_CACHE_PATH)

    print(f"--- Extracting embeddings using {config.ENCODER_MODEL} ---")
    
    # 2. Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(config.ENCODER_MODEL)
    model = AutoModel.from_pretrained(config.ENCODER_MODEL)
    
    # 3. Freeze the model (Fine-tuning is NOT allowed)
    for param in model.parameters():
        param.requires_grad = False
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    embeddings = []

    # 4. Process in batches to avoid memory errors
    batch_size = 16 
    for i in tqdm(range(0, len(text_list), batch_size)):
        batch_text = text_list[i : i + batch_size]
        
        # Tokenize
        inputs = tokenizer(batch_text, padding=True, truncation=True, 
                           max_length=128, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            
            # Extract [CLS] token (the first token in the sequence)
            # This represents the contextual embedding for the whole sentence
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls_embeddings)

    # 5. Combine and Cache the results (Requirement: Part B)
    final_embeddings = np.vstack(embeddings)
    np.save(config.EMBEDDING_CACHE_PATH, final_embeddings)
    print("--- Embeddings saved to disk! ---")
    
    return final_embeddings