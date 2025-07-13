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

The `cattle_data.csv` file contains:
- **Cattle lot management data** with weight tracking and feed information
- **Realistic data quality issues** for testing the cleaning system
- **Three categories of data problems**:
  1. **Outlier weights** (>1.3x normal range) - should be deleted
  2. **Questionable weights** (1.0-1.3x normal range) - need manual review
  3. **Incorrect ready_to_load labels** - should be corrected

### Data Schema
- `lot_id`: Unique identifier for each cattle lot
- `entry_weight_lbs`: Weight when cattle entered the feedlot (lbs)
- `exit_weight_lbs`: Weight when ready for processing (lbs)
- `days_on_feed`: Number of days in the feedlot
- `breed`: Cattle breed (Angus, Hereford, Charolais)
- `feed_type`: Type of feed used (Corn, Mixed)
- `ready_to_load`: Whether the lot is ready for processing (Yes/No)

### Expected Weight Ranges
- **Normal entry weight**: 550-800 lbs
- **Normal exit weight**: 1100-1500 lbs
- **Outlier threshold**: >1.3x normal range (>1040 lbs entry, >1950 lbs exit)
- **Review threshold**: 1.0-1.3x normal range

### Data Quality Issues in Sample
- **LOT005, LOT012**: Extreme outlier weights (should be deleted)
- **LOT022, LOT029**: Questionable weights (need manual review)
- **LOT003, LOT007, LOT009, etc.**: Incorrect ready_to_load labels (should be corrected)

You can use this file to test the cattle data cleaning agent:

```bash
# Command line usage
python main.py --requirements "Clean cattle weight data and fix ready_to_load labels" --data-source "data/input/cattle_data.csv"

# Web interface usage
1. Start the web server: python app.py
2. Upload the cattle_data.csv file
3. Describe your cleaning requirements: "Process cattle data for outliers and label corrections"
```

## Notes

- The `.gitkeep` files ensure that empty directories are tracked by Git
- Temporary files in `temp/` are excluded from Git tracking
- Always backup your original data before processing

