import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--deploy_ratio", type=float, default=0.10)
    parser.add_argument("--train_out", type=str, required=True)
    parser.add_argument("--val_out", type=str, required=True)
    parser.add_argument("--test_out", type=str, required=True)
    parser.add_argument("--deploy_out", type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load dataset
    df = pd.read_parquet(args.data)
    print(f"Loaded {len(df)} rows")
    
    # Split: train (60%) vs temp (40%)
    train_df, temp_df = train_test_split(
        df,
        test_size=(1 - args.train_ratio),
        random_state=args.seed,
        shuffle=True
    )
    
    # Split temp into val, test, deploy
    # val = 15% of original, test = 15%, deploy = 10%
    # From temp (40%):
    #   val = 15/40 = 37.5% of temp
    #   test = 15/40 = 37.5% of temp
    #   deploy = 10/40 = 25% of temp
    
    # First split: separate deploy (25% of temp)
    val_test_df, deploy_df = train_test_split(
        temp_df,
        test_size=(args.deploy_ratio / (1 - args.train_ratio)),
        random_state=args.seed,
        shuffle=True
    )
    
    # Split remaining into val and test (50/50)
    val_df, test_df = train_test_split(
        val_test_df,
        test_size=0.5,
        random_state=args.seed,
        shuffle=True
    )
    
    # Write outputs
    os.makedirs(args.train_out, exist_ok=True)
    os.makedirs(args.val_out, exist_ok=True)
    os.makedirs(args.test_out, exist_ok=True)
    os.makedirs(args.deploy_out, exist_ok=True)
    
    train_df.to_parquet(os.path.join(args.train_out, "data.parquet"))
    val_df.to_parquet(os.path.join(args.val_out, "data.parquet"))
    test_df.to_parquet(os.path.join(args.test_out, "data.parquet"))
    deploy_df.to_parquet(os.path.join(args.deploy_out, "data.parquet"))
    
    print(f"Train rows: {len(train_df)}")
    print(f"Validation rows: {len(val_df)}")
    print(f"Test rows: {len(test_df)}")
    print(f"Deployment rows: {len(deploy_df)}")

if __name__ == "__main__":
    main()