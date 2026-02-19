#!/usr/bin/env python3
"""
Script to download data from Kaggle competition: American Express Default Prediction

Before running this script, you need to set up Kaggle API credentials:
1. Go to https://www.kaggle.com/ and sign in
2. Go to Account settings (https://www.kaggle.com/settings)
3. Scroll down to "API" section and click "Create New Token"
4. This will download a kaggle.json file
5. Place the kaggle.json file in ~/.kaggle/ directory:
   - Create the directory if it doesn't exist: mkdir -p ~/.kaggle
   - Move the file: mv ~/Downloads/kaggle.json ~/.kaggle/
   - Set proper permissions: chmod 600 ~/.kaggle/kaggle.json
"""

import os
import sys
import zipfile
from pathlib import Path

def check_kaggle_credentials():
    """Check if Kaggle credentials are configured."""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"
    
    if not kaggle_json.exists():
        print("=" * 70)
        print("ERROR: Kaggle credentials not found!")
        print("=" * 70)
        print("\nTo download data from Kaggle, you need to set up API credentials:")
        print("\n1. Go to https://www.kaggle.com/ and sign in")
        print("2. Go to Account settings: https://www.kaggle.com/settings")
        print("3. Scroll down to 'API' section and click 'Create New Token'")
        print("4. This will download a kaggle.json file")
        print(f"5. Place the kaggle.json file in {kaggle_dir}/")
        print("   Run these commands:")
        print(f"   mkdir -p {kaggle_dir}")
        print(f"   mv ~/Downloads/kaggle.json {kaggle_dir}/")
        print(f"   chmod 600 {kaggle_dir}/kaggle.json")
        print("\nAfter setting up credentials, run this script again.")
        print("=" * 70)
        return False
    
    # Check permissions
    if os.stat(kaggle_json).st_mode & 0o077 != 0:
        print("WARNING: kaggle.json has incorrect permissions.")
        print(f"Run: chmod 600 {kaggle_json}")
        return False
    
    return True

def download_competition_data():
    """Download data from Kaggle competition."""
    # Check credentials first before importing kaggle (which tries to authenticate on import)
    if not check_kaggle_credentials():
        return False
    
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("ERROR: Kaggle package not installed.")
        print("Install it with: pip install kaggle")
        return False
    
    # Set up paths
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data" / "external"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading data to: {data_dir}")
    print("This may take a while depending on your internet connection...")
    
    try:
        api = KaggleApi()
        api.authenticate()
        
        competition_name = "amex-default-prediction"
        print(f"\nDownloading competition data: {competition_name}")
        api.competition_download_files(competition_name, path=str(data_dir))
        
        # Unzip all zip files in the directory
        print("\nExtracting zip files...")
        zip_files = list(data_dir.glob("*.zip"))
        for zip_file in zip_files:
            print(f"  Extracting {zip_file.name}...")
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            # Optionally remove the zip file after extraction
            # zip_file.unlink()
        
        print("\n" + "=" * 70)
        print("SUCCESS: Data downloaded and extracted successfully!")
        print(f"Data location: {data_dir}")
        print("=" * 70)
        return True
        
    except Exception as e:
        print("\n" + "=" * 70)
        print("ERROR: Failed to download data")
        print("=" * 70)
        print(f"Error message: {str(e)}")
        print("\nPossible issues:")
        print("1. Check your internet connection")
        print("2. Verify your Kaggle credentials are correct")
        print("3. Make sure you have accepted the competition rules on Kaggle")
        print("4. Check if the competition name is correct")
        print("=" * 70)
        return False

if __name__ == "__main__":
    success = download_competition_data()
    sys.exit(0 if success else 1)

