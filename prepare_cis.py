"""
prepare_cis_fraud.py

Downloads the IEEE-CIS Fraud Detection dataset via the Kaggle API, merges the
identity and transaction tables, and engineers a multi-class target variable 
to evaluate Pandora's Regret (grouping by operational audit costs).

Usage:
    # Ensure you have your kaggle.json set up in ~/.kaggle/
    pip install kaggle pandas
    python prepare_cis_fraud.py --output /workspace/data/ieee-cis
"""

import os
import zipfile
import argparse
import pandas as pd
from pathlib import Path
import subprocess

# ── Constants ─────────────────────────────────────────────────────────────────
KAGGLE_COMPETITION = "ieee-fraud-detection"

# Pandora's Regret Class Mapping
# 0 = Legitimate (Cost: $0)
# 1 = Promo/Friendly Fraud (Cost: $) -> ProductCD in ['C', 'S']
# 2 = Card-Not-Present (Cost: $$) -> ProductCD == 'W'
# 3 = Identity Theft (Cost: $$$$) -> ProductCD in ['H', 'R']
CLASS_MAP = {
    0: "Legitimate",
    1: "Promo_Friendly_Fraud",
    2: "Card_Not_Present",
    3: "Identity_Theft"
}

def download_and_extract(download_dir: Path):
    """Download dataset using Kaggle API and extract."""
    download_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading IEEE-CIS dataset to {download_dir}...")
    subprocess.run([
        "kaggle", "competitions", "download", 
        "-c", KAGGLE_COMPETITION, 
        "-p", str(download_dir)
    ], check=True)
    
    zip_path = download_dir / f"{KAGGLE_COMPETITION}.zip"
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(download_dir)
    print("Extraction complete.")

def create_pandora_classes(df: pd.DataFrame) -> pd.DataFrame:
    """Map binary isFraud to multi-class Pandora targets based on ProductCD."""
    def map_target(row):
        if row['isFraud'] == 0:
            return 0  # Legitimate
        
        prod = str(row.get('ProductCD', ''))
        if prod in ['C', 'S']:
            return 1  # Promo/Friendly Fraud
        elif prod == 'W':
            return 2  # Card-Not-Present
        elif prod in ['H', 'R']:
            return 3  # Identity Theft
        return 1 # Default fallback for fraud

    df['pandora_target'] = df.apply(map_target, axis=1)
    df['pandora_class_name'] = df['pandora_target'].map(CLASS_MAP)
    return df

def prepare_data(output_dir: Path, download_dir: Path):
    """Merge tables, create classes, and perform time-based split."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not (download_dir / "train_transaction.csv").exists():
        download_and_extract(download_dir)

    print("Loading raw CSVs...")
    train_txn = pd.read_csv(download_dir / "train_transaction.csv")
    train_id = pd.read_csv(download_dir / "train_identity.csv")

    print("Merging Transaction and Identity tables...")
    # Left merge as not all transactions have identity info
    df = train_txn.merge(train_id, on='TransactionID', how='left')

    print("Mapping targets for Pandora's Regret...")
    df = create_pandora_classes(df)
    
    # Sort by time (TransactionDT is a timedelta from a reference datetime)
    df = df.sort_values('TransactionDT').reset_index(drop=True)

    print("Performing time-based split (70% Train, 15% Val, 15% Test)...")
    n = len(df)
    train_idx = int(n * 0.70)
    val_idx = int(n * 0.85)

    train_df = df.iloc[:train_idx]
    val_df = df.iloc[train_idx:val_idx]
    test_df = df.iloc[val_idx:]

    print(f"\nSplit Summary:")
    print(f"  Train: {len(train_df)} rows")
    print(f"  Val:   {len(val_df)} rows")
    print(f"  Test:  {len(test_df)} rows")

    # Save outputs
    print(f"\nSaving splits to {output_dir}...")
    train_df.to_csv(output_dir / "train_pandora.csv", index=False)
    val_df.to_csv(output_dir / "val_pandora.csv", index=False)
    test_df.to_csv(output_dir / "test_pandora.csv", index=False)
    
    print("\nClass Distribution in Test Set:")
    print(test_df['pandora_class_name'].value_counts())
    print("\nData preparation complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare IEEE-CIS for Pandora's Regret.")
    parser.add_argument("--output", type=Path, default=Path("./data/ieee-cis-processed"))
    parser.add_argument("--download_dir", type=Path, default=Path("./data/ieee-cis-raw"))
    args = parser.parse_args()
    
    prepare_data(args.output, args.download_dir)
