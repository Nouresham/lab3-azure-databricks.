import argparse
import os
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=str, required=True)
    parser.add_argument("--sentiment", type=str, required=True)
    parser.add_argument("--tfidf", type=str, required=True)
    parser.add_argument("--sbert", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("=" * 50)
    print("MERGING ALL FEATURES")
    print("=" * 50)
    
    # Load all feature datasets
    print("\n📊 Loading features...")
    
    length_df = pd.read_parquet(os.path.join(args.length, "data.parquet"))
    print(f"  - Length features: {length_df.shape}")
    
    sentiment_df = pd.read_parquet(os.path.join(args.sentiment, "data.parquet"))
    print(f"  - Sentiment features: {sentiment_df.shape}")
    
    tfidf_df = pd.read_parquet(os.path.join(args.tfidf, "data.parquet"))
    print(f"  - TF-IDF features: {tfidf_df.shape}")
    
    sbert_df = pd.read_parquet(os.path.join(args.sbert, "data.parquet"))
    print(f"  - SBERT embeddings: {sbert_df.shape}")
    
    # Get identifier columns
    id_columns = ['asin', 'reviewerID']
    
    # Verify all datasets have the same identifiers
    for df, name in [(length_df, 'Length'), (sentiment_df, 'Sentiment'), 
                     (tfidf_df, 'TF-IDF'), (sbert_df, 'SBERT')]:
        missing_cols = [col for col in id_columns if col not in df.columns]
        if missing_cols:
            print(f"⚠️ {name} missing columns: {missing_cols}")
            # Try to find alternative identifier columns
            available_ids = [col for col in df.columns if col in ['asin', 'reviewerID', 'reviewerID', 'product_id']]
            if available_ids:
                print(f"   Using available IDs: {available_ids}")
    
    # Start with length features (which should have the base identifiers)
    print("\n🔗 Merging features...")
    
    # First, ensure we have the base identifiers
    merged_df = length_df.copy()
    
    # Merge sentiment features
    merge_cols = id_columns + [col for col in sentiment_df.columns if col not in id_columns + ['reviewText', 'normalized_text']]
    merged_df = merged_df.merge(
        sentiment_df[merge_cols],
        on=[col for col in id_columns if col in sentiment_df.columns],
        how='inner'
    )
    print(f"  - After sentiment merge: {merged_df.shape}")
    
    # Merge TF-IDF features
    tfidf_cols = [col for col in tfidf_df.columns if col not in id_columns]
    merged_df = merged_df.merge(
        tfidf_df,
        on=[col for col in id_columns if col in tfidf_df.columns],
        how='inner'
    )
    print(f"  - After TF-IDF merge: {merged_df.shape}")
    
    # Merge SBERT embeddings
    sbert_cols = [col for col in sbert_df.columns if col not in id_columns]
    merged_df = merged_df.merge(
        sbert_df,
        on=[col for col in id_columns if col in sbert_df.columns],
        how='inner'
    )
    print(f"  - After SBERT merge: {merged_df.shape}")
    
    # Display feature breakdown
    print("\n📋 FEATURE SUMMARY:")
    print(f"  - Total rows: {len(merged_df)}")
    print(f"  - Total columns: {len(merged_df.columns)}")
    
    length_features = [col for col in merged_df.columns if 'length' in col]
    sentiment_features = [col for col in merged_df.columns if 'sentiment' in col]
    tfidf_features = [col for col in merged_df.columns if 'tfidf_' in col]
    bert_features = [col for col in merged_df.columns if 'bert_embedding_' in col]
    id_columns_present = [col for col in id_columns if col in merged_df.columns]
    
    print(f"\n  Feature types:")
    print(f"    - Identifier columns: {len(id_columns_present)} ({', '.join(id_columns_present)})")
    print(f"    - Length features: {len(length_features)}")
    print(f"    - Sentiment features: {len(sentiment_features)}")
    print(f"    - TF-IDF features: {len(tfidf_features)}")
    print(f"    - BERT embeddings: {len(bert_features)}")
    print(f"    - Total features: {len(length_features) + len(sentiment_features) + len(tfidf_features) + len(bert_features)}")
    
    # Save merged dataset
    print("\n💾 Saving merged features...")
    os.makedirs(args.out, exist_ok=True)
    merged_df.to_parquet(os.path.join(args.out, "data.parquet"))
    
    # Save feature metadata
    feature_metadata = {
        'id_columns': id_columns_present,
        'length_features': length_features,
        'sentiment_features': sentiment_features,
        'tfidf_features': tfidf_features,
        'bert_features': bert_features,
        'total_features': len(length_features) + len(sentiment_features) + len(tfidf_features) + len(bert_features),
        'total_rows': len(merged_df),
        'shape': list(merged_df.shape)
    }
    
    import json
    with open(os.path.join(args.out, "feature_metadata.json"), "w") as f:
        json.dump(feature_metadata, f, indent=2)
    
    print(f"\n✅ Merged features saved to: {args.out}")
    print(f"   Final dataset: {merged_df.shape[0]} rows, {merged_df.shape[1]} columns")
    
    # Show sample
    print("\n📝 Sample of merged features (first row):")
    sample_row = merged_df.iloc[0]
    print(f"  IDs: asin={sample_row.get('asin', 'N/A')}, reviewerID={sample_row.get('reviewerID', 'N/A')}")
    if length_features:
        print(f"  Length: chars={sample_row.get('review_length_chars', 'N/A')}, words={sample_row.get('review_length_words', 'N/A')}")
    if sentiment_features:
        print(f"  Sentiment: compound={sample_row.get('sentiment_compound', 'N/A'):.3f}")
    if tfidf_features:
        tfidf_sample = [col for col in tfidf_features[:3] if sample_row[col] > 0]
        if tfidf_sample:
            print(f"  Top TF-IDF features: {tfidf_sample}")
    if bert_features:
        print(f"  BERT embeddings: [{sample_row[bert_features[0]]:.3f}, {sample_row[bert_features[1]]:.3f}, ...]")

if __name__ == "__main__":
    main()