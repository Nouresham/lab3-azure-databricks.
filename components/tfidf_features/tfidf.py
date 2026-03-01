import argparse
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--val", type=str, required=True)
    parser.add_argument("--test", type=str, required=True)
    parser.add_argument("--train_out", type=str, required=True)
    parser.add_argument("--val_out", type=str, required=True)
    parser.add_argument("--test_out", type=str, required=True)
    parser.add_argument("--max_features", type=int, default=500)
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load datasets
    train_df = pd.read_parquet(args.train)
    val_df = pd.read_parquet(args.val)
    test_df = pd.read_parquet(args.test)
    
    print(f"Train rows: {len(train_df)}")
    print(f"Validation rows: {len(val_df)}")
    print(f"Test rows: {len(test_df)}")
    
    # Find text column
    text_col = None
    for col in ['normalized_text', 'reviewText', 'text']:
        if col in train_df.columns:
            text_col = col
            break
    
    if text_col is None:
        raise ValueError("No text column found in dataset")
    
    print(f"Using '{text_col}' for TF-IDF")
    
    # Get identifier columns
    id_columns = []
    for col in ['asin', 'reviewerID']:
        if col in train_df.columns:
            id_columns.append(col)
    
    # Initialize TF-IDF vectorizer
    print(f"Fitting TF-IDF with max_features={args.max_features}...")
    vectorizer = TfidfVectorizer(
        max_features=args.max_features,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.8
    )
    
    # Fit on training data only (prevent data leakage)
    train_texts = train_df[text_col].fillna('').astype(str)
    vectorizer.fit(train_texts)
    
    # Transform all splits
    print("Transforming training data...")
    train_tfidf = vectorizer.transform(train_texts)
    
    print("Transforming validation data...")
    val_tfidf = vectorizer.transform(val_df[text_col].fillna('').astype(str))
    
    print("Transforming test data...")
    test_tfidf = vectorizer.transform(test_df[text_col].fillna('').astype(str))
    
    # Convert to DataFrame with feature names
    feature_names = vectorizer.get_feature_names_out()
    
    train_tfidf_df = pd.DataFrame(
        train_tfidf.toarray(),
        columns=[f"tfidf_{name}" for name in feature_names]
    )
    val_tfidf_df = pd.DataFrame(
        val_tfidf.toarray(),
        columns=[f"tfidf_{name}" for name in feature_names]
    )
    test_tfidf_df = pd.DataFrame(
        test_tfidf.toarray(),
        columns=[f"tfidf_{name}" for name in feature_names]
    )
    
    # Add identifier columns
    for col in id_columns:
        train_tfidf_df[col] = train_df[col].values
        val_tfidf_df[col] = val_df[col].values
        test_tfidf_df[col] = test_df[col].values
    
    # Reorder columns to put identifiers first
    cols = id_columns + [c for c in train_tfidf_df.columns if c not in id_columns]
    train_tfidf_df = train_tfidf_df[cols]
    val_tfidf_df = val_tfidf_df[cols]
    test_tfidf_df = test_tfidf_df[cols]
    
    # Save vectorizer for later use
    os.makedirs(args.train_out, exist_ok=True)
    with open(os.path.join(args.train_out, "tfidf_vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)
    
    # Save outputs
    train_tfidf_df.to_parquet(os.path.join(args.train_out, "data.parquet"))
    val_tfidf_df.to_parquet(os.path.join(args.val_out, "data.parquet"))
    test_tfidf_df.to_parquet(os.path.join(args.test_out, "data.parquet"))
    
    print(f"✅ TF-IDF features created:")
    print(f"  - {len(feature_names)} features")
    print(f"  - Train shape: {train_tfidf_df.shape}")
    print(f"  - Validation shape: {val_tfidf_df.shape}")
    print(f"  - Test shape: {test_tfidf_df.shape}")
    print(f"\nSample features: {feature_names[:10]}")

if __name__ == "__main__":
    main()