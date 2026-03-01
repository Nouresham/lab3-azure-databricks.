import argparse
import os
import pandas as pd
import re
import string

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    return parser.parse_args()

def normalize_text(text):
    """Normalize review text"""
    if pd.isna(text) or text == "":
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove numbers (optional - keep if you think they're important)
    text = re.sub(r'\d+', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def main():
    args = parse_args()
    
    # Load dataset
    df = pd.read_parquet(args.data)
    
    print(f"Loaded {len(df)} rows")
    print(f"Columns: {df.columns.tolist()}")
    
    # Check if reviewText column exists
    if 'reviewText' not in df.columns:
        # Try to find text column
        text_columns = [col for col in df.columns if 'review' in col.lower() or 'text' in col.lower()]
        if text_columns:
            text_col = text_columns[0]
            print(f"Using '{text_col}' as text column")
        else:
            raise ValueError("No text column found in dataset")
    else:
        text_col = 'reviewText'
    
    # Apply normalization
    print("Normalizing text...")
    df['normalized_text'] = df[text_col].apply(normalize_text)
    
    # Filter out empty or very short reviews (<10 characters)
    original_count = len(df)
    df = df[df['normalized_text'].str.len() >= 10]
    filtered_count = len(df)
    print(f"Removed {original_count - filtered_count} rows with very short or empty text")
    
    # Keep original text and normalized text
    # Also keep identifier columns
    id_columns = ['asin', 'reviewerID'] if 'asin' in df.columns and 'reviewerID' in df.columns else []
    
    # Create output dataframe
    output_cols = id_columns + [text_col, 'normalized_text'] + ['overall'] if 'overall' in df.columns else id_columns + [text_col, 'normalized_text']
    output_df = df[output_cols] if all(col in df.columns for col in output_cols) else df
    
    # Write output
    os.makedirs(args.out, exist_ok=True)
    output_df.to_parquet(os.path.join(args.out, "data.parquet"))
    
    print(f"✅ Saved {len(output_df)} normalized reviews")
    print(f"Sample normalized text: {output_df['normalized_text'].iloc[0][:100]}...")

if __name__ == "__main__":
    main()
    