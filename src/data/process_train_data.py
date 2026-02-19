#!/usr/bin/env python3
"""
Process train data and separate labels from features.

This script:
1. Loads the training data from data/external/train.parquet
2. Identifies the target column (label)
3. Separates features (X) and labels (y)
4. Saves the processed data to data/processed/
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def load_train_data(data_path):
    """Load training data from parquet file."""
    print(f"Loading training data from {data_path}...")
    df = pd.read_parquet(data_path)
    print(f"Data shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    return df

def check_for_labels_file(external_dir):
    """Check if there's a separate labels file."""
    possible_label_files = [
        'train_labels.csv',
        'train_labels.parquet',
        'train_labels.csv.gz',
        'labels.csv',
        'labels.parquet'
    ]
    
    for label_file in possible_label_files:
        label_path = external_dir / label_file
        if label_path.exists():
            return label_path
    
    return None

def identify_target_column(df, external_dir=None):
    """Identify the target column in the dataset."""
    # First check if there's a separate labels file
    if external_dir:
        labels_file = check_for_labels_file(external_dir)
        if labels_file:
            print(f"Found separate labels file: {labels_file}")
            return labels_file  # Return path to indicate it's a separate file
    
    # Common target column names in Kaggle competitions
    possible_targets = ['target', 'default', 'label', 'y']
    
    for col in possible_targets:
        if col in df.columns:
            return col
    
    # If no common name found, check for binary columns (0/1)
    # But exclude feature columns (D_, S_, P_, B_, R_)
    feature_prefixes = ['D_', 'S_', 'P_', 'B_', 'R_']
    binary_cols = []
    for col in df.columns:
        # Skip feature columns and ID columns
        if any(col.startswith(p) for p in feature_prefixes) or col in ['customer_ID', 'id', 'customer_id']:
            continue
        unique_vals = df[col].unique()
        if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0, -1, -1.0}):
            # Check if it's a reasonable target distribution (not too imbalanced)
            val_counts = df[col].value_counts()
            if len(val_counts) == 2:
                min_pct = min(val_counts) / len(df)
                if 0.01 <= min_pct <= 0.5:  # Reasonable target distribution
                    binary_cols.append(col)
    
    if len(binary_cols) == 1:
        return binary_cols[0]
    
    # If still not found, return None
    print("\nWarning: Could not automatically identify target column.")
    print("Note: This appears to be feature-only data. Target may be in a separate file.")
    return None

def separate_features_labels(df, target_col, external_dir=None):
    """Separate features and labels from the dataframe."""
    # Get customer ID if it exists (usually first column)
    id_col = None
    if 'customer_ID' in df.columns:
        id_col = 'customer_ID'
    elif 'id' in df.columns:
        id_col = 'id'
    elif 'customer_id' in df.columns:
        id_col = 'customer_id'
    
    # Handle separate labels file
    if target_col and isinstance(target_col, Path):
        # Load labels from separate file
        labels_df = pd.read_parquet(target_col) if target_col.suffix == '.parquet' else pd.read_csv(target_col)
        # Assume labels are in the first column or match by index
        if id_col:
            # Try to merge by ID
            if id_col in labels_df.columns:
                y = labels_df.set_index(id_col).iloc[:, 0]
                ids = df[id_col].copy()
                # Align labels with features by ID
                y = y.reindex(df[id_col].values)
                y = y.values
            else:
                # Assume same order
                y = labels_df.iloc[:, 0].values
                ids = df[id_col].copy() if id_col else None
        else:
            # Assume same order
            y = labels_df.iloc[:, 0].values
            ids = None
    elif target_col and target_col in df.columns:
        # Target is in the same dataframe
        y = df[target_col].copy()
        exclude_cols = [target_col]
        if id_col:
            exclude_cols.append(id_col)
        X = df.drop(columns=exclude_cols).copy()
        ids = df[id_col].copy() if id_col else None
        return X, y, ids
    else:
        # No target found - just process features
        print("Warning: No target column found. Processing features only.")
        exclude_cols = []
        if id_col:
            exclude_cols.append(id_col)
        X = df.drop(columns=exclude_cols).copy() if exclude_cols else df.copy()
        y = None
        ids = df[id_col].copy() if id_col else None
        return X, y, ids
    
    # Separate features (exclude ID columns)
    exclude_cols = []
    if id_col:
        exclude_cols.append(id_col)
    
    X = df.drop(columns=exclude_cols).copy()
    
    # Convert y to Series if it's an array
    if isinstance(y, np.ndarray):
        y = pd.Series(y, name='target')
    
    return X, y, ids

def save_processed_data(X, y, ids, output_dir):
    """Save processed features and labels to parquet files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving processed data to {output_dir}...")
    
    # Save features
    features_path = output_dir / "X_train.parquet"
    X.to_parquet(features_path, index=False, compression='snappy')
    print(f"  Features saved: {features_path} ({X.shape})")
    
    # Save labels if they exist
    labels_path = None
    if y is not None:
        labels_path = output_dir / "y_train.parquet"
        y.to_frame().to_parquet(labels_path, index=False, compression='snappy')
        print(f"  Labels saved: {labels_path} ({y.shape})")
    else:
        print("  Note: No labels to save (target column not found)")
    
    # Save IDs if they exist
    ids_path = None
    if ids is not None:
        ids_path = output_dir / "customer_ids.parquet"
        ids.to_frame().to_parquet(ids_path, index=False, compression='snappy')
        print(f"  Customer IDs saved: {ids_path} ({ids.shape})")
    
    return features_path, labels_path, ids_path

def print_data_summary(X, y, ids):
    """Print summary statistics of the processed data."""
    print("\n" + "=" * 70)
    print("DATA SUMMARY")
    print("=" * 70)
    print(f"\nFeatures (X):")
    print(f"  Shape: {X.shape}")
    print(f"  Columns: {len(X.columns)}")
    print(f"  Memory usage: {X.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"  Data types:")
    print(X.dtypes.value_counts())
    
    if y is not None:
        print(f"\nLabels (y):")
        print(f"  Shape: {y.shape}")
        print(f"  Value counts:")
        print(y.value_counts().sort_index())
        print(f"  Class distribution:")
        print(y.value_counts(normalize=True).sort_index())
    else:
        print(f"\nLabels (y): Not available")
    
    if ids is not None:
        print(f"\nCustomer IDs:")
        print(f"  Shape: {ids.shape}")
        print(f"  Unique IDs: {ids.nunique()}")
    
    print("=" * 70)

def main():
    """Main function to process train data."""
    # Set up paths
    data_dir = project_root / "data"
    external_dir = data_dir / "external"
    processed_dir = data_dir / "processed"
    
    train_path = external_dir / "train.parquet"
    
    # Check if train data exists
    if not train_path.exists():
        print(f"ERROR: Training data not found at {train_path}")
        print("Please ensure train.parquet exists in data/external/")
        sys.exit(1)
    
    try:
        # Load data
        df = load_train_data(train_path)
        
        # Identify target column
        print("\nIdentifying target column...")
        target_col = identify_target_column(df, external_dir)
        
        if target_col:
            if isinstance(target_col, Path):
                print(f"Target found in separate file: {target_col}")
            else:
                print(f"Target column identified: '{target_col}'")
        else:
            print("\nNote: No target column found. Will process features only.")
        
        # Separate features and labels
        print("\nSeparating features and labels...")
        X, y, ids = separate_features_labels(df, target_col, external_dir)
        
        # Print summary
        print_data_summary(X, y, ids)
        
        # Save processed data
        save_processed_data(X, y, ids, processed_dir)
        
        print("\n" + "=" * 70)
        print("SUCCESS: Train data processed and saved!")
        print("=" * 70)
        print(f"\nProcessed files:")
        print(f"  Features: data/processed/X_train.parquet")
        if y is not None:
            print(f"  Labels: data/processed/y_train.parquet")
        if ids is not None:
            print(f"  IDs: data/processed/train_ids.parquet")
        
        return True
        
    except Exception as e:
        print("\n" + "=" * 70)
        print("ERROR: Failed to process train data")
        print("=" * 70)
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

