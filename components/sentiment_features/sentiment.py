import argparse
import os
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download VADER lexicon (first time only)
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

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
    
    # Find text column
    text_col = None
    for col in ['normalized_text', 'reviewText', 'text']:
        if col in df.columns:
            text_col = col
            break
    
    if text_col is None:
        raise ValueError("No text column found in dataset")
    
    print(f"Using '{text_col}' for sentiment analysis")
    
    # Initialize VADER
    sia = SentimentIntensityAnalyzer()
    
    # Apply sentiment analysis
    print("Calculating sentiment scores...")
    
    sentiments = []
    for text in df[text_col].fillna(''):
        if text.strip():
            scores = sia.polarity_scores(text)
            sentiments.append(scores)
        else:
            sentiments.append({'neg': 0.0, 'neu': 0.0, 'pos': 0.0, 'compound': 0.0})
    
    # Extract sentiment components
    df['sentiment_neg'] = [s['neg'] for s in sentiments]
    df['sentiment_neu'] = [s['neu'] for s in sentiments]
    df['sentiment_pos'] = [s['pos'] for s in sentiments]
    df['sentiment_compound'] = [s['compound'] for s in sentiments]
    
    # Keep identifier columns and features
    id_columns = ['asin', 'reviewerID'] if 'asin' in df.columns and 'reviewerID' in df.columns else []
    
    output_cols = id_columns + ['sentiment_neg', 'sentiment_neu', 'sentiment_pos', 'sentiment_compound']
    output_df = df[output_cols]
    
    # Write output
    os.makedirs(args.out, exist_ok=True)
    output_df.to_parquet(os.path.join(args.out, "data.parquet"))
    
    print(f"✅ Saved sentiment features for {len(output_df)} reviews")
    print(f"Sample sentiment:")
    print(f"  - Negative: {output_df['sentiment_neg'].iloc[0]:.3f}")
    print(f"  - Neutral: {output_df['sentiment_neu'].iloc[0]:.3f}")
    print(f"  - Positive: {output_df['sentiment_pos'].iloc[0]:.3f}")
    print(f"  - Compound: {output_df['sentiment_compound'].iloc[0]:.3f}")

if __name__ == "__main__":
    main()