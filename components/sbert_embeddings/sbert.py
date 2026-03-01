import argparse
import os
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import pickle

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="all-MiniLM-L6-v2")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load dataset
    df = pd.read_parquet(args.data)
    
    print(f"Loaded {len(df)} rows")
    
    # Find text column
    text_col = None
    for col in ['normalized_text', 'reviewText', 'text']:
        if col in df.columns:
            text_col = col
            break
    
    if text_col is None:
        raise ValueError("No text column found in dataset")
    
    print(f"Using '{text_col}' for embeddings")
    print(f"Loading SBERT model: {args.model_name}...")
    
    # Load model
    model = SentenceTransformer(args.model_name)
    
    # Get identifier columns
    id_columns = []
    for col in ['asin', 'reviewerID']:
        if col in df.columns:
            id_columns.append(col)
    
    # Generate embeddings in batches
    texts = df[text_col].fillna('').astype(str).tolist()
    
    print(f"Generating embeddings for {len(texts)} texts...")
    batch_size = 64
    embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_embeddings = model.encode(batch_texts, show_progress_bar=False)
        embeddings.append(batch_embeddings)
        print(f"  Processed {min(i+batch_size, len(texts))}/{len(texts)}")
    
    embeddings = np.vstack(embeddings)
    
    # Create DataFrame with embeddings
    embedding_dim = embeddings.shape[1]
    embedding_cols = [f"bert_embedding_{i}" for i in range(embedding_dim)]
    
    embedding_df = pd.DataFrame(embeddings, columns=embedding_cols)
    
    # Add identifier columns
    for col in id_columns:
        embedding_df[col] = df[col].values
    
    # Reorder columns to put identifiers first
    cols = id_columns + embedding_cols
    embedding_df = embedding_df[cols]
    
    # Save model for later use
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "sbert_model.pkl"), "wb") as f:
        pickle.dump(model, f)
    
    # Save embeddings
    embedding_df.to_parquet(os.path.join(args.out, "data.parquet"))
    
    print(f"✅ SBERT embeddings created:")
    print(f"  - Model: {args.model_name}")
    print(f"  - Embedding dimension: {embedding_dim}")
    print(f"  - Shape: {embedding_df.shape}")
    print(f"  - Sample embedding (first 5 values): {embedding_df[embedding_cols[0]].iloc[0][:5]}")

if __name__ == "__main__":
    main()