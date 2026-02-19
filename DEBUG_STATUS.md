# Debug Status - AMEX Default Prediction Data Processing

## Current State

### ✅ Successfully Processed:
1. **X_train.parquet** (1.7GB) - Features extracted (5,531,451 rows × 189 columns)
2. **customer_ids.parquet** (41MB) - Customer IDs extracted (5,531,451 rows)

### ❌ Missing:
1. **y_train.parquet** - Binary default indicator (target labels)

## Issue Analysis

### Data Structure:
- **train.parquet**: Contains time-series data
  - 5,531,451 total rows (statements)
  - 458,913 unique customers
  - ~12.1 statements per customer on average
  - Contains: customer_ID + 189 feature columns
  - **Does NOT contain target labels**

### Why Target is Missing:
The AMEX competition provides target labels in a **separate file**: `train_labels.csv`
- This file contains: `customer_ID, target`
- Target is binary: 0 (no default) or 1 (default)
- Target is at **customer level**, not statement level

### What Needs to Happen:
1. Download `train_labels.csv` from Kaggle
2. Process it to create `y_train.parquet` with binary default indicator
3. The labels need to be aligned with the customer IDs

## Files Created:
- `src/data/process_labels.py` - Script to process train_labels.csv when available
- `src/data/process_train_data.py` - Script to process features (already run)

## Next Steps:
1. Download train_labels.csv from Kaggle:
   ```bash
   kaggle competitions download -c amex-default-prediction -f train_labels.csv
   ```
2. Place it in `data/external/train_labels.csv`
3. Run: `python3 src/data/process_labels.py`
4. This will create `data/processed/y_train.parquet` with the binary default indicator

## Note:
The file `customer_ids.parquet` contains customer IDs (hash strings), NOT the binary default indicator. The binary indicator will be in `y_train.parquet` once the labels are processed.

