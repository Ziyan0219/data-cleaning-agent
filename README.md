# 🐄 Cattle Data Cleaning Agent

An AI-powered data quality control system specifically designed for cattle weight data and ready-to-load status validation. This system uses intelligent classification to automatically handle data quality issues while maintaining human oversight for critical decisions.

## 🎯 Key Features

### **Intelligent Data Classification**
- **🔴 Extreme Outliers**: Automatically removes impossible weight data (>1.3x normal threshold)
- **🟡 Suspicious Data**: Flags questionable weights for manual review (1.0-1.3x threshold)  
- **🟢 Label Corrections**: Automatically fixes obvious labeling errors

### **Business Logic**
- **Weight Thresholds**: 500-1500 lbs normal range, >1950 lbs extreme outliers
- **Smart Processing**: Balances automation with human oversight
- **Quality Scoring**: Comprehensive data quality improvement metrics

### **Production Ready**
- **Web Interface**: User-friendly FastAPI web application
- **Detailed Reports**: Comprehensive processing reports with actionable insights
- **Error Handling**: Robust error handling and logging
- **Scalable Architecture**: Modular design for easy extension

## 🚀 Quick Start

### **Method 1: Web Interface (Recommended)**

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.template .env
# Edit .env and add your OpenAI API key

# Start web server
python app_v2.py

# Open browser to http://localhost:8000
```

### **Method 2: Command Line**

```bash
# Test with sample data
python test_system.py

# Process custom data
python main.py --requirements "Process cattle data for weight validation" --data-source "path/to/your/data.csv"
```

## 📁 Project Structure

```
data_cleaning_agent_english/
├── 📊 data/                          # Data directory
│   ├── input/                        # Input data files
│   │   ├── cattle_data.csv          # Sample cattle data with quality issues
│   │   └── sample_data.csv          # General sample data
│   ├── output/                       # Cleaned data output
│   └── temp/                         # Temporary processing files
├── 🤖 src/                           # Source code
│   ├── agents/                       # AI agents
│   │   ├── main_controller_fixed.py  # Working main controller
│   │   ├── data_analysis_agent.py    # Data analysis agent
│   │   ├── data_cleaning_agent.py    # Data cleaning agent
│   │   ├── quality_validation_agent.py # Quality validation agent
│   │   └── result_aggregation_agent.py # Result aggregation agent
│   ├── config/                       # Configuration management
│   │   └── settings.py              # Application settings
│   ├── schemas/                      # Data schemas
│   │   └── state.py                 # State definitions
│   └── utils/                        # Utility functions
│       └── json_utils.py            # JSON serialization utilities
├── 🌐 Web Interface
│   ├── app_v2.py                    # Working FastAPI web application
│   └── app.py                       # Original web application
├── 🧪 Testing
│   ├── test_system.py               # System test script
│   └── main.py                      # Command line interface
├── 📚 Documentation
│   ├── README.md                    # This file
│   └── data_cleaning_agent_english_guide.md # Detailed development guide
└── ⚙️ Configuration
    ├── requirements.txt             # Python dependencies
    ├── .env.template               # Environment variables template
    └── .gitignore                  # Git ignore rules
```

## 🐄 Sample Data Format

The system expects CSV files with the following columns:

```csv
lot_id,entry_weight_lbs,exit_weight_lbs,days_on_feed,breed,feed_type,ready_to_load
LOT001,650,1250,180,Angus,Corn,Yes
LOT002,680,1320,185,Hereford,Mixed,Yes
LOT003,720,1180,175,Charolais,Corn,No
```

### **Data Quality Issues Handled**
1. **Extreme outliers**: LOT005 (2100 lbs entry) → Automatically deleted
2. **Suspicious weights**: LOT022 (1850 lbs) → Flagged for manual review  
3. **Label errors**: Normal weight but marked "No" → Auto-corrected to "Yes"

## 📊 Processing Logic

### **Three-Tier Classification System**

```
🔴 EXTREME OUTLIERS (>1950 lbs)
├── Action: Automatic deletion
├── Rationale: Impossible weights for cattle
└── Report: Lists all deleted lots with weights

🟡 SUSPICIOUS WEIGHTS (1500-1950 lbs)  
├── Action: Flag for manual review
├── Rationale: Possible but requires verification
└── Report: Lists lots needing human inspection

🟢 LABEL ERRORS (Normal weight + "No" status)
├── Action: Automatic correction to "Yes"
├── Rationale: Clear data entry errors
└── Report: Lists all corrected labels
```

## 📋 Sample Report Output

```
🐄 CATTLE DATA CLEANING REPORT
==================================================
📅 Generated: 2025-07-13 12:27:17

📊 EXECUTIVE SUMMARY
--------------------
• Original lots: 30
• Processed lots: 26  
• Removed lots: 4
• Quality improvement: 100.0%

🔍 PROBLEM CLASSIFICATION
-------------------------
1. EXTREME OUTLIERS (Deleted): 4 lots
   • LOT005: Entry=2100lbs, Exit=2800lbs
   • LOT012: Entry=3200lbs, Exit=4100lbs

2. SUSPICIOUS WEIGHTS (Manual Review): 0 lots
   • No suspicious weights found

3. LABEL ERRORS (Auto-corrected): 8 lots
   • LOT003: Changed from 'no' to 'Yes'
   • LOT007: Changed from 'no' to 'Yes'

⚙️ PROCESSING ACTIONS
--------------------
✂️ DELETED 4 lots: Weight exceeds 1.3x normal threshold
🔧 CORRECTED 8 labels: Normal weight but incorrectly marked as not ready

🎯 NEXT STEPS
------------
• No manual review required
• All remaining lots are ready for processing

📈 DATA QUALITY METRICS
-----------------------
• Ready to load: 26/26 lots (100.0%)
• Data completeness: 100.0%
• Weight range compliance: 26/26 lots
```

## 🛠️ Installation & Setup

### **Prerequisites**
- Python 3.8+
- OpenAI API key

### **Installation Steps**

1. **Clone the repository**
   ```bash
   git clone https://github.com/Ziyan0219/data-cleaning-agent.git
   cd data-cleaning-agent
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**
   ```bash
   cp .env.template .env
   # Edit .env file and add your OpenAI API key:
   # OPENAI_API_KEY=sk-your-actual-api-key-here
   ```

4. **Test the system**
   ```bash
   python test_system.py
   ```

5. **Start web interface**
   ```bash
   python app_v2.py
   # Open http://localhost:8000 in your browser
   ```

## 🔧 Configuration

### **Environment Variables (.env)**
```bash
# Required: OpenAI API Configuration
OPENAI_API_KEY=sk-your-actual-api-key-here

# Optional: Additional API Keys
ANTHROPIC_API_KEY=your-anthropic-key-if-needed
```

### **Weight Thresholds (Customizable)**
```python
# In src/agents/main_controller_fixed.py
weight_thresholds = {
    "min_normal": 500,    # Minimum normal weight (lbs)
    "max_normal": 1500,   # Maximum normal weight (lbs)  
    "extreme_threshold": 1950  # Extreme outlier threshold (lbs)
}
```

## 🧪 Testing

### **System Test**
```bash
python test_system.py
```

### **Web Interface Test**
```bash
python app_v2.py
# Navigate to http://localhost:8000/test
```

### **Custom Data Test**
```bash
python main.py --requirements "Your cleaning requirements" --data-source "path/to/your/data.csv"
```

## 📈 Performance Metrics

- **Processing Speed**: ~0.02 seconds for 30 lots
- **Quality Improvement**: Up to 100% issue resolution
- **Accuracy**: Intelligent classification with minimal false positives
- **Scalability**: Handles datasets from 10s to 1000s of records

## 🔍 Troubleshooting

### **Common Issues**

1. **Import Errors**
   ```bash
   # Solution: Use the working version
   python app_v2.py  # Instead of app.py
   python test_system.py  # For testing
   ```

2. **API Key Issues**
   ```bash
   # Check your .env file
   cat .env
   # Ensure OPENAI_API_KEY is set correctly
   ```

3. **Data Format Issues**
   ```bash
   # Ensure CSV has required columns:
   # lot_id, entry_weight_lbs, exit_weight_lbs, ready_to_load
   ```

### **Debug Mode**
```bash
# Enable detailed logging
export LOG_LEVEL=DEBUG
python test_system.py
```

## 🚀 Deployment

### **Local Development**
```bash
python app_v2.py
# Access at http://localhost:8000
```

### **Production Deployment**
```bash
# Using uvicorn
uvicorn app_v2:app --host 0.0.0.0 --port 8000

# Using gunicorn
gunicorn app_v2:app -w 4 -k uvicorn.workers.UvicornWorker
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Contact

If you have questions or suggestions, please contact us through:

- Submit Issue: [GitHub Issues](https://github.com/Ziyan0219/data-cleaning-agent/issues)
- Email: ziyanxinbci@gmail.com

## 🏆 Acknowledgments

- Built with LangChain and FastAPI
- Designed for agricultural data quality control
- Optimized for cattle weight validation workflows

---

**Version 2.1.0** - Simplified & Reliable | Production Ready 🎉

