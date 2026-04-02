lab3 and lab 4


Lab 3 – Data Preprocessing with Azure Databricks
Overview

ETL pipeline using Azure Databricks and Apache Spark on Amazon Electronics review data. Follows the Medallion Architecture:

Bronze: Raw JSON data

Silver => Cleaned and enriched data

Gold =>  Curated dataset for analytics ttechnologies

Apache Spark and Databricks

ADLS Gen2 for storage

Parquet format

Databricks Jobs to automate ETL

Python (pandas, matplotlib, seaborn)

nsights

Most ratings are 4–5 stars

Ratings dip in 2004–2005, then rise

Longer reviews often more detailed
-------------------
lab 4
# Lab 4: Text Feature Engineering with Azure ML

## Project Overview
Feature engineering pipeline for Amazon Electronics reviews (300,000 samples).

## Features Created
1. **Length Features**: `review_length_chars`, `review_length_words`
   - Why: Captures review verbosity, longer reviews often more detailed

2. **Sentiment Features**: `sentiment_neg`, `sentiment_neu`, `sentiment_pos`, `sentiment_compound`
   - Why: Captures emotional tone using VADER

3. **TF-IDF Features**: 500 most important unigrams/bigrams
   - Why: Captures word importance and frequency

4. **BERT Embeddings**: 384-dimensional vectors from Sentence-BERT
   - Why: Captures semantic meaning beyond individual words

## Pipeline Components
- `split_dataset`: 70/15/15 train/validation/test split
- `normalize_text`: Lowercase, remove URLs/numbers/punctuation
- `length_features`: Character and word counts
- `sentiment_features`: VADER sentiment analysis
- `tfidf_features`: TF-IDF vectorization
- `sbert_embeddings`: BERT embeddings
- `merge_features`: Combines all features

## How to Run
1. Register datastore: `az ml datastore create --file datastores/curated_datastore.yml`
2. Register components: `az ml component create --file components/*/component.yml`
3. Submit pipeline: `az ml job create --file pipelines/feature_pipeline.yml`

## Results
- Pipeline completed successfully: `silver_nut_9xqsq9mfhn`
- Final feature dataset URI: `azureml:azureml_ddf14590-8bd8-4e9c-8e73-d8a9aeda2d75_output_data_out:1`
