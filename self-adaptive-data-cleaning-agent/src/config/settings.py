"""
Configuration Management Module

This module manages all application configuration parameters,
including LLM configuration, Agent configuration, data processing configuration, etc.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class LLMConfig:
    """LLM Configuration"""
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    api_key: str = ""
    temperature: float = 0.0
    max_tokens: int = 4000
    timeout: int = 30
    max_retries: int = 3
    
    def __post_init__(self):
        if not self.api_key:
            self.api_key = os.getenv("OPENAI_API_KEY", "")


@dataclass
class AgentConfig:
    """Agent Configuration"""
    max_workers: int = 4
    timeout_seconds: int = 300
    retry_attempts: int = 3
    cache_enabled: bool = True
    parallel_execution: bool = True
    
    # Agent-specific configurations
    analysis_agent_config: Dict[str, Any] = field(default_factory=dict)
    cleaning_agent_config: Dict[str, Any] = field(default_factory=dict)
    validation_agent_config: Dict[str, Any] = field(default_factory=dict)
    aggregation_agent_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataConfig:
    """Data Processing Configuration"""
    max_file_size_mb: int = 100
    supported_formats: list = field(default_factory=lambda: [
        "csv", "xlsx", "json", "parquet"
    ])
    chunk_size: int = 10000
    memory_limit_mb: int = 1000
    temp_dir: str = "data/temp"
    output_dir: str = "data/output"
    backup_enabled: bool = True


@dataclass
class QualityConfig:
    """Quality Control Configuration"""
    # Quality thresholds
    min_completeness: float = 0.8
    min_accuracy: float = 0.9
    min_consistency: float = 0.85
    min_validity: float = 0.9
    min_uniqueness: float = 0.95
    
    # Validation configuration
    enable_statistical_validation: bool = True
    enable_schema_validation: bool = True
    enable_business_rule_validation: bool = True
    
    # Anomaly detection
    outlier_detection_method: str = "iqr"  # iqr, zscore, isolation_forest
    outlier_threshold: float = 3.0


@dataclass
class LoggingConfig:
    """Logging Configuration"""
    level: str = "INFO"
    file_path: str = "logs/agent.log"
    max_file_size_mb: int = 10
    backup_count: int = 5
    format: str = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name} | {message}"
    
    def __post_init__(self):
        self.level = os.getenv("LOG_LEVEL", self.level)
        self.file_path = os.getenv("LOG_FILE", self.file_path)


@dataclass
class DatabaseConfig:
    """Database Configuration"""
    url: str = ""
    pool_size: int = 5
    max_overflow: int = 10
    pool_timeout: int = 30
    
    def __post_init__(self):
        self.url = os.getenv("DATABASE_URL", self.url)


class Settings:
    """Application Settings"""
    
    def __init__(self):
        self.llm = LLMConfig()
        self.agent = AgentConfig()
        self.data = DataConfig()
        self.quality = QualityConfig()
        self.logging = LoggingConfig()
        self.database = DatabaseConfig()
        
        # Load configuration from environment variables
        self._load_from_env()
    
    def _load_from_env(self):
        """Load configuration from environment variables"""
        # Agent configuration
        self.agent.max_workers = int(os.getenv("PARALLEL_WORKERS", "4"))
        self.agent.timeout_seconds = int(os.getenv("TIMEOUT_SECONDS", "300"))
        self.agent.retry_attempts = int(os.getenv("MAX_RETRIES", "3"))
        self.agent.cache_enabled = os.getenv("CACHE_ENABLED", "true").lower() == "true"
        
        # Data configuration
        self.data.max_file_size_mb = int(os.getenv("MAX_FILE_SIZE_MB", "100"))
        self.data.chunk_size = int(os.getenv("CHUNK_SIZE", "10000"))
        
        # Quality configuration
        self.quality.min_completeness = float(os.getenv("MIN_COMPLETENESS", "0.8"))
        self.quality.min_accuracy = float(os.getenv("MIN_ACCURACY", "0.9"))
    
    def get_llm_config(self, provider: Optional[str] = None) -> LLMConfig:
        """Get LLM configuration"""
        if provider and provider != self.llm.provider:
            # Create provider-specific configuration
            config = LLMConfig()
            config.provider = provider
            
            if provider == "anthropic":
                config.api_key = os.getenv("ANTHROPIC_API_KEY", "")
                config.model = "claude-3-sonnet-20240229"
            elif provider == "google":
                config.api_key = os.getenv("GOOGLE_API_KEY", "")
                config.model = "gemini-pro"
            
            return config
        
        return self.llm
    
    def get_cleaning_rules(self) -> Dict[str, Any]:
        """Get cleaning rules configuration"""
        return {
            "missing_values": {
                "strategies": ["drop", "fill_mean", "fill_median", "fill_mode", "interpolate"],
                "default_strategy": "fill_mean",
                "threshold": 0.5  # Missing value ratio threshold
            },
            "duplicates": {
                "detection_method": "exact_match",  # exact_match, fuzzy_match
                "keep": "first",  # first, last, none
                "similarity_threshold": 0.9
            },
            "outliers": {
                "detection_methods": ["iqr", "zscore", "isolation_forest"],
                "default_method": "iqr",
                "threshold": 3.0,
                "action": "flag"  # remove, flag, transform
            },
            "format_standardization": {
                "date_formats": ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"],
                "phone_formats": ["international", "national"],
                "email_validation": True,
                "case_normalization": "lower"
            },
            "data_validation": {
                "schema_validation": True,
                "range_validation": True,
                "pattern_validation": True,
                "business_rule_validation": True
            }
        }
    
    def get_prompt_templates(self) -> Dict[str, str]:
        """Get prompt templates"""
        return {
            "data_analysis": """
You are a professional data quality analysis expert. Please analyze the following data sample and identify quality issues:

Data sample:
{data_sample}

Data schema:
{schema_info}

Please provide the following analysis:
1. Data quality issue identification
2. Issue severity assessment
3. Cleaning strategy recommendations
4. Expected improvement effects

Please return results in JSON format.
""",
            
            "data_cleaning": """
You are a data cleaning expert. Please perform data cleaning based on the following information:

Raw data:
{raw_data}

Quality issues:
{quality_issues}

Cleaning strategy:
{cleaning_strategy}

Please execute cleaning operations and return:
1. Cleaned data
2. List of operations performed
3. Improvement effect assessment

Please ensure data format consistency.
""",
            
            "quality_validation": """
You are a data quality validation expert. Please validate the following cleaning results:

Original data:
{original_data}

Cleaned data:
{cleaned_data}

Cleaning log:
{cleaning_log}

Please assess:
1. Cleaning quality score
2. Potential issue identification
3. Improvement recommendations
4. Whether re-cleaning is needed

Please return validation results in JSON format.
"""
        }
    
    def validate_config(self) -> bool:
        """Validate configuration validity"""
        errors = []
        
        # Validate API key
        if not self.llm.api_key:
            errors.append("Missing LLM API key")
        
        # Validate directories
        for directory in [self.data.temp_dir, self.data.output_dir]:
            if not os.path.exists(directory):
                try:
                    os.makedirs(directory, exist_ok=True)
                except Exception as e:
                    errors.append(f"Cannot create directory {directory}: {e}")
        
        # Validate quality thresholds
        quality_metrics = [
            self.quality.min_completeness,
            self.quality.min_accuracy,
            self.quality.min_consistency,
            self.quality.min_validity,
            self.quality.min_uniqueness
        ]
        
        for metric in quality_metrics:
            if not 0 <= metric <= 1:
                errors.append(f"Quality metric must be between 0 and 1: {metric}")
        
        if errors:
            print("Configuration validation errors:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        return True


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get settings instance"""
    return settings


def reload_settings():
    """Reload settings"""
    global settings
    settings = Settings()
    return settings

