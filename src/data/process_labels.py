#!/usr/bin/env python3
"""
Process train labels and create y_train.parquet with binary default indicator.

This script:
1. Loads train_labels.csv from data/external/ (or data/raw/)
2. Extracts the binary target (default indicator)
3. Saves to data/processed/y_train.parquet
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def find_labels_file():
    """Find the train labels file."""
    possible_locations = [
        project_root / "data" / "external" / "train_labels.csv",
        project_root / "data" / "external" / "train_labels.csv",
        project_root / "data" / "raw" / "train_labels.csv",
        project_root / "data" / "raw" / "train_labels.csv.gz",
    ]
    
    for path in possible_locations:
        if path.exists():
            return path
    
    return None

def process_labels():
    """Process train labels and save to processed folder."""
    # Find labels file
    labels_file = find_labels_file()
    
    if labels_file is None:
        print("=" * 70)
        print("ERROR: train_labels file not found!")
        print("=" * 70)
        print("\nThe train labels file should contain the binary default indicator.")
        print("Please download train_labels.csv from Kaggle and place it in:")
        print("  - data/external/train_labels.csv")
        print("  - or data/raw/train_labels.csv")
        print("\nTo download from Kaggle:")
        print("  kaggle competitions download -c amex-default-prediction -f train_labels.csv")
        print("=" * 70)
        return False
    
    print(f"Found labels file: {labels_file}")
    
    # Load labels
    print("Loading labels...")
    if labels_file.suffix == '.gz':
        labels_df = pd.read_csv(labels_file, compression='gzip')
    else:
        labels_df = pd.read_csv(labels_file)
    
    print(f"Labels shape: {labels_df.shape}")
    print(f"Columns: {list(labels_df.columns)}")
    
    # Identify target column
    target_col = None
    for col in ['target', 'default', 'label']:
        if col in labels_df.columns:
            target_col = col
            break
    
    if target_col is None:
        # Assume first column after customer_ID is the target
        if 'customer_ID' in labels_df.columns:
            target_col = labels_df.columns[1] if len(labels_df.columns) > 1 else None
        else:
            target_col = labels_df.columns[0]
    
    if target_col is None:
        print("ERROR: Could not identify target column in labels file")
        return False
    
    print(f"Target column: '{target_col}'")
    
    # Extract target
    y = labels_df[target_col].copy()
    
    # Verify it's binary
    unique_vals = sorted(y.unique())
    print(f"Unique values: {unique_vals}")
    print(f"Value counts:\n{y.value_counts().sort_index()}")
    print(f"Class distribution:\n{y.value_counts(normalize=True).sort_index()}")
    
    if not set(unique_vals).issubset({0, 1, 0.0, 1.0}):
        print("WARNING: Target column is not binary (0/1)")
    
    # Save to processed folder
    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    labels_path = processed_dir / "y_train.parquet"
    y.to_frame(name='target').to_parquet(labels_path, index=False, compression='snappy')
    
    print(f"\n✓ Labels saved to: {labels_path}")
    print(f"  Shape: {y.shape}")
    print(f"  Size: {labels_path.stat().st_size / 1024**2:.2f} MB")
    
    return True

if __name__ == "__main__":
    success = process_labels()
    sys.exit(0 if success else 1)

