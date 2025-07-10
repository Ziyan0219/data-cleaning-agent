# Data Directory

This directory contains data files for the Data Cleaning Agent system.

## Directory Structure

```
data/
├── input/          # Input data files (CSV, Excel, JSON)
│   ├── sample_data.csv    # Sample dataset for testing
│   └── .gitkeep          # Keeps directory in Git
├── output/         # Cleaned data output files
│   └── .gitkeep          # Keeps directory in Git
└── temp/           # Temporary processing files
    └── .gitkeep          # Keeps directory in Git
```

## Usage

### Input Directory (`input/`)
- Place your raw data files here for processing
- Supported formats: CSV, Excel (.xlsx, .xls), JSON
- Example: `sample_data.csv` - A sample employee dataset with common data quality issues

### Output Directory (`output/`)
- Cleaned data files will be saved here after processing
- Files are automatically named with timestamps and session IDs
- Format: `cleaned_data_<session_id>.csv`

### Temp Directory (`temp/`)
- Used for temporary files during processing
- Files are automatically cleaned up after processing
- Do not manually place files here

## Sample Data

The `sample_data.csv` file contains:
- Employee information with missing email addresses
- Mixed data types and formats
- Common data quality issues for testing the cleaning system

You can use this file to test the data cleaning agent:

```bash
# Command line usage
python main.py --requirements "Fill missing emails and standardize formats" --data-source "data/input/sample_data.csv"

# Web interface usage
1. Start the web server: python app.py
2. Upload the sample_data.csv file
3. Describe your cleaning requirements
```

## Notes

- The `.gitkeep` files ensure that empty directories are tracked by Git
- Temporary files in `temp/` are excluded from Git tracking
- Always backup your original data before processing

