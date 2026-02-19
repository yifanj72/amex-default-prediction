# American Express Default Prediction

## Project Overview

This project focuses on predicting credit card default risk for American Express customers using machine learning techniques.

## Problem Statement

Predict the probability that a customer will default on their credit card payment, enabling American Express to:
- Assess credit risk more accurately
- Make informed decisions on credit limits and approvals
- Reduce financial losses from defaults
- Improve customer risk management

## Dataset

The dataset is from the [American Express Default Prediction competition on Kaggle](https://www.kaggle.com/competitions/amex-default-prediction).

**Data Location:** `data/external/`

**Files:**
- `train.parquet` (1.5GB) - Training data with integer dtypes (post-processed)
- `test.parquet` (3.1GB) - Test data with integer dtypes (post-processed)

**Data Format:** Parquet files with integer dtypes (post-processed from original competition data where float to int type conversions were done)

**Note:** The data files are stored in the `data/external/` directory and are not tracked by git due to their large size.

## Methodology

[Add methodology details here]

## Project Structure

```
amex-default-prediction/
├── data/
│   ├── raw/           # Raw data files
│   ├── processed/     # Processed/cleaned data
│   └── external/      # External data sources
├── notebooks/          # Jupyter notebooks for exploration and analysis
├── src/                # Source code
│   ├── data/          # Data processing scripts
│   ├── features/      # Feature engineering
│   ├── models/        # Model training and evaluation
│   └── utils/         # Utility functions
├── models/             # Saved model files
├── reports/            # Generated reports and visualizations
├── requirements.txt    # Python dependencies
└── README.md          # Project documentation
```

## Technologies Used

- **Language:** Python
- **Libraries:** 
  - pandas, numpy (Data manipulation)
  - scikit-learn (Machine Learning)
  - matplotlib, seaborn (Visualization)
  - jupyter (Notebooks)

## Getting Started

### Prerequisites

- Python 3.8+
- pip or conda

### Installation

```bash
# Clone the repository
git clone https://github.com/yifanj72/amex-default-prediction.git
cd amex-default-prediction

# Install dependencies
pip install -r requirements.txt
```

## Usage

[Add usage instructions here]

## Results

[Add results and findings here]

## Future Improvements

[Add future work and improvements here]

## Author

Yifan Jiang

## License

[Add license information here]

