import argparse
import os
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load dataset
    df = pd.read_parquet(args.data)
    
    print(f"Loaded {len(df)} rows")
    print(f"Columns: {df.columns.tolist()}")
    
    # Find text column
    text_col = None
    for col in ['normalized_text', 'reviewText', 'text']:
        if col in df.columns:
            text_col = col
            break
    
    if text_col is None:
        raise ValueError("No text column found in dataset")
    
    print(f"Using '{text_col}' for length features")
    
    # Create length features
    df['review_length_chars'] = df[text_col].str.len()
    df['review_length_words'] = df[text_col].str.split().str.len()
    
    # Keep identifier columns and features
    id_columns = ['asin', 'reviewerID'] if 'asin' in df.columns and 'reviewerID' in df.columns else []
    
    output_cols = id_columns + ['review_length_chars', 'review_length_words']
    output_df = df[output_cols]
    
    # Write output
    os.makedirs(args.out, exist_ok=True)
    output_df.to_parquet(os.path.join(args.out, "data.parquet"))
    
    print(f"✅ Saved length features for {len(output_df)} reviews")
    print(f"Stats:")
    print(f"  - Avg chars: {output_df['review_length_chars'].mean():.1f}")
    print(f"  - Avg words: {output_df['review_length_words'].mean():.1f}")
    print(f"  - Max chars: {output_df['review_length_chars'].max()}")
    print(f"  - Max words: {output_df['review_length_words'].max()}")

if __name__ == "__main__":
    main()