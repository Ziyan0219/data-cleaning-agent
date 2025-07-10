# Data Cleaning Agent System

An intelligent data cleaning system based on LangChain and LangGraph, featuring a multi-agent collaborative architecture that can automatically identify data quality issues and execute intelligent cleaning operations.

## ✨ Features

- 🤖 **Multi-Agent Collaborative Architecture** - Specialized division of labor with efficient collaboration
- 🔍 **Intelligent Data Quality Analysis** - Deep analysis combining rule engines and LLM
- 🛠️ **Automated Data Cleaning** - Support for multiple cleaning strategies and custom rules
- ✅ **Quality Validation & Assessment** - Comprehensive quality metrics calculation and validation
- 📊 **Detailed Report Generation** - Complete records of cleaning process and results
- 🔧 **Highly Configurable** - Flexible configuration options to adapt to different needs
- 📈 **Performance Monitoring** - Built-in monitoring and logging system

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the project
git clone https://github.com/Ziyan0219/data-cleaning-agent.git
cd data-cleaning-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy environment variable template
cp .env.template .env

# Edit .env file and fill in your API key
# OPENAI_API_KEY=sk-your-openai-api-key-here
```

### 3. Test Configuration

```bash
# Test system configuration
python main.py --test-config
```

### 4. Run Example

```bash
# Run basic example
python examples/basic_usage.py

# Or use command line interface
python main.py --requirements "Handle missing values and duplicates" --data-source "data/input/sample.csv"
```

## 📖 Usage

### Web Interface (Recommended)

The easiest way to use the system is through the web interface:

```bash
# Start the web server
python app.py

# Or using uvicorn directly
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Then open your browser and go to: **http://localhost:8000**

**Web Interface Features:**
- 📁 **File Upload**: Support for CSV, Excel, and JSON files
- 📝 **Requirements Input**: Describe your cleaning needs in natural language
- 📊 **Real-time Results**: View quality metrics and processing status
- 💾 **Download Results**: Get cleaned data and detailed reports
- 🎯 **User-friendly**: No coding required

### Command Line Interface

```bash
# Basic usage
python main.py --requirements "Cleaning requirements description" --data-source "data file path"

# Specify output file
python main.py -r "Handle missing values" -d "input.csv" -o "cleaned_output.csv"

# Use different model
python main.py --model "gpt-4" --requirements "Complex cleaning task" --data-source "data.csv"
```

### Python API

```python
import asyncio
from src.agents.main_controller import process_cleaning_request

async def clean_data():
    result = await process_cleaning_request(
        user_requirements="Clean customer data, handle missing values and duplicates",
        data_source="data/input/customers.csv"
    )
    
    if result['status'] == 'completed':
        print(f"Cleaning completed, quality score: {result['quality_metrics']}")
    else:
        print(f"Cleaning failed: {result['error']}")

# Run
asyncio.run(clean_data())
```

### Advanced Configuration

```python
from src.config.settings import get_settings

# Get settings instance
settings = get_settings()

# Customize quality thresholds
settings.quality.min_completeness = 0.9
settings.quality.min_accuracy = 0.95

# Customize cleaning rules
custom_rules = settings.get_cleaning_rules()
custom_rules["missing_values"]["default_strategy"] = "interpolate"
```

## 🏗️ Architecture Design

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Cleaning Agent System               │
├─────────────────────────────────────────────────────────────┤
│                    Coordination Layer (Supervisor)          │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              Main Controller Agent                      │ │
│  └─────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                    Execution Layer (Worker Agents)          │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────┐ │
│  │ Data        │ │ Cleaning    │ │ Quality     │ │ Result  │ │
│  │ Analysis    │ │ Execution   │ │ Validation  │ │ Aggreg. │ │
│  │ Agent       │ │ Agent       │ │ Agent       │ │ Agent   │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────┘ │
├─────────────────────────────────────────────────────────────┤
│                    Tools Layer (Tools & Utilities)          │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────┐ │
│  │ Data        │ │ Rule        │ │ LLM         │ │ Storage │ │
│  │ Connectors  │ │ Engine      │ │ Service     │ │ Manager │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
data_cleaning_agent_english/
├── src/                          # Source code
│   ├── agents/                   # Agent implementations
│   │   ├── main_controller.py    # Main Controller Agent
│   │   ├── data_analysis_agent.py # Data Analysis Agent
│   │   └── ...                   # Other Agents
│   ├── config/                   # Configuration management
│   ├── schemas/                  # Data schemas
│   ├── tools/                    # Tool modules
│   └── utils/                    # Utilities
├── data/                         # Data directory
│   ├── input/                    # Input data
│   ├── output/                   # Output data
│   └── temp/                     # Temporary files
├── examples/                     # Example code
├── tests/                        # Test files
├── logs/                         # Log files
├── main.py                       # Main program entry
├── requirements.txt              # Dependencies
└── README.md                     # Project documentation
```

## 🔧 Configuration Options

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `OPENAI_MODEL` | Model to use | `gpt-4o-mini` |
| `LOG_LEVEL` | Log level | `INFO` |
| `MAX_RETRIES` | Maximum retry attempts | `3` |
| `TIMEOUT_SECONDS` | Timeout duration | `30` |

### Quality Thresholds

```python
# Configure in src/config/settings.py
class QualityConfig:
    min_completeness: float = 0.8    # Completeness threshold
    min_accuracy: float = 0.9        # Accuracy threshold
    min_consistency: float = 0.85    # Consistency threshold
    min_validity: float = 0.9        # Validity threshold
    min_uniqueness: float = 0.95     # Uniqueness threshold
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific tests
pytest tests/test_agents.py

# Run integration tests
pytest tests/test_integration.py -v
```

## 📊 Monitoring and Logging

### Log Configuration

Log files are located in the `logs/` directory, supporting the following levels:
- `DEBUG`: Detailed debugging information
- `INFO`: General information
- `WARNING`: Warning messages
- `ERROR`: Error messages

### Performance Monitoring

The system includes built-in performance monitoring to track:
- Execution time
- Memory usage
- API call count
- Quality metric changes

## 🚀 Deployment

### Docker Deployment

```bash
# Build image
docker build -t data-cleaning-agent .

# Run container
docker run -d \
  -e OPENAI_API_KEY=your-api-key \
  -v $(pwd)/data:/app/data \
  -p 8000:8000 \
  data-cleaning-agent
```

### Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f
```

## 🔌 Extension Development

### Adding Custom Agents

```python
from src.agents.base_agent import BaseAgent

class CustomAgent(BaseAgent):
    def __init__(self, llm):
        super().__init__(llm)
    
    def custom_processing(self, data: str) -> Dict:
        # Implement custom logic
        pass
```

### Adding Custom Cleaning Rules

```python
# Add new rules in configuration
custom_rules = {
    "custom_validation": {
        "phone_format": r"^\+?1?\d{9,15}$",
        "email_domains": ["company.com", "partner.org"]
    }
}
```

## 🐛 Troubleshooting

### Q: API key error
**A:** Check if `OPENAI_API_KEY` is correctly set in the `.env` file

### Q: Out of memory
**A:** Adjust `chunk_size` configuration parameter or enable streaming processing

### Q: Slow processing
**A:** Consider using a faster model or enabling parallel processing

For more issues, see [Troubleshooting Guide](docs/troubleshooting.md)

## 📚 Documentation

- [Complete Development Guide](docs/development_guide.md)
- [API Reference](docs/api_reference.md)
- [Configuration Guide](docs/configuration.md)
- [Extension Development](docs/extensions.md)

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the project
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## 🙏 Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) - Powerful LLM application framework
- [LangGraph](https://github.com/langchain-ai/langgraph) - State graph workflow engine
- [OpenAI](https://openai.com/) - Excellent LLM services

## 📞 Contact

If you have questions or suggestions, please contact us through:

- Submit Issue: [GitHub Issues](https://github.com/Ziyan0219/data-cleaning-agent/issues)
- Email: ziyanxinbci@gmail.com

---

⭐ If this project helps you, please give us a Star!

