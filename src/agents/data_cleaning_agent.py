"""
Data Cleaning Agent Implementation

This module implements the data cleaning agent responsible for executing
actual data cleaning operations based on the analysis results and execution plan.
"""

import json
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd
import numpy as np
from loguru import logger

from langchain_core.messages import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate

from ..config.settings import get_settings


class DataCleaningAgent:
    """Data Cleaning Agent Class"""
    
    def __init__(self, llm):
        self.llm = llm
        self.settings = get_settings()
        self.cleaning_prompt = self._create_cleaning_prompt()
        
        # Cleaning strategy registry
        self.cleaning_strategies = {
            "missing_values": self._handle_missing_values,
            "duplicates": self._handle_duplicates,
            "format_standardization": self._standardize_formats,
            "outlier_treatment": self._treat_outliers,
            "data_validation": self._validate_data_integrity,
            "type_conversion": self._convert_data_types
        }
        
        # Strategy mapping: maps specific cleaning methods to strategy categories
        self.strategy_mapping = {
            # Missing values strategies
            "fill_mean": "missing_values",
            "fill_median": "missing_values",
            "fill_mode": "missing_values",
            "fill_forward": "missing_values",
            "fill_backward": "missing_values",
            "fill_zero": "missing_values",
            "fill_constant": "missing_values",
            "drop_missing": "missing_values",
            "interpolate": "missing_values",
            
            # Duplicate strategies
            "remove_duplicates": "duplicates",
            "keep_first": "duplicates",
            "keep_last": "duplicates",
            "mark_duplicates": "duplicates",
            
            # Format standardization strategies
            "standardize_text": "format_standardization",
            "normalize_case": "format_standardization",
            "trim_whitespace": "format_standardization",
            "standardize_dates": "format_standardization",
            "standardize_phone": "format_standardization",
            "standardize_email": "format_standardization",
            
            # Outlier treatment strategies
            "remove_outliers": "outlier_treatment",
            "cap_outliers": "outlier_treatment",
            "transform_outliers": "outlier_treatment",
            "manual_review": "outlier_treatment",
            
            # Data validation strategies
            "validate_range": "data_validation",
            "validate_format": "data_validation",
            "validate_constraints": "data_validation",
            "check_consistency": "data_validation",
            
            # Type conversion strategies
            "convert_numeric": "type_conversion",
            "convert_datetime": "type_conversion",
            "convert_categorical": "type_conversion",
            "convert_boolean": "type_conversion"
        }
        
        logger.info("Data Cleaning Agent initialized successfully")
    
    def _create_cleaning_prompt(self) -> ChatPromptTemplate:
        """Create cleaning prompt template"""
        template = self.settings.get_prompt_templates()["data_cleaning"]
        
        return ChatPromptTemplate.from_messages([
            ("system", """You are a data cleaning expert. Your task is to execute data cleaning operations based on the provided information and return cleaned data with operation logs.

Please ensure:
1. Data format consistency
2. Preserve data integrity
3. Document all operations performed
4. Provide quality improvement assessment"""),
            ("human", template)
        ])
    
    def execute_cleaning_plan(self, data: str, execution_plan: List[Dict], 
                            config: Dict) -> Dict[str, Any]:
        """Execute cleaning plan"""
        logger.info("Starting cleaning plan execution")
        
        try:
            # Parse data
            df = self._parse_data_to_dataframe(data)
            if df is None:
                raise ValueError("Unable to parse input data")
            
            # Initialize tracking
            operation_log = []
            current_data = df.copy()
            
            # Execute operations in order
            for operation in execution_plan:
                try:
                    result = self._execute_single_operation(current_data, operation, config)
                    current_data = result["data"]
                    operation_log.append(result["log"])
                    
                    logger.info(f"Completed operation: {operation.get('operation_type', 'unknown')}")
                    
                except Exception as e:
                    logger.error(f"Operation failed: {operation.get('operation_type', 'unknown')} - {str(e)}")
                    operation_log.append({
                        "operation_id": operation.get("operation_id", "unknown"),
                        "operation_type": operation.get("operation_type", "unknown"),
                        "status": "failed",
                        "error": str(e),
                        "timestamp": datetime.now()
                    })
            
            # Convert back to string format
            cleaned_data = self._dataframe_to_string(current_data, data)
            
            # Calculate improvement metrics
            improvement_metrics = self._calculate_improvement_metrics(df, current_data)
            
            result = {
                "cleaned_data": cleaned_data,
                "operation_log": operation_log,
                "improvement_metrics": improvement_metrics,
                "completed_operations": [op["operation_id"] for op in operation_log if op.get("status") == "success"],
                "failed_operations": [op["operation_id"] for op in operation_log if op.get("status") == "failed"],
                "cleaning_summary": {
                    "total_operations": len(execution_plan),
                    "successful_operations": len([op for op in operation_log if op.get("status") == "success"]),
                    "failed_operations": len([op for op in operation_log if op.get("status") == "failed"]),
                    "data_quality_improvement": improvement_metrics.get("overall_improvement", 0)
                }
            }
            
            logger.info("Cleaning plan execution completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Cleaning plan execution failed: {str(e)}")
            return {
                "cleaned_data": data,  # Return original data on failure
                "operation_log": [],
                "improvement_metrics": {},
                "completed_operations": [],
                "failed_operations": [],
                "cleaning_summary": {"error": str(e)},
                "error": str(e)
            }
    
    def _execute_single_operation(self, data: pd.DataFrame, operation: Dict, 
                                config: Dict) -> Dict[str, Any]:
        """Execute a single cleaning operation"""
        operation_type = operation.get("issue_type", operation.get("operation_type", "unknown"))
        operation_id = operation.get("operation_id", str(uuid.uuid4()))
        
        logger.debug(f"Executing operation: {operation_type}")
        
        try:
            # Get strategy function with mapping support
            strategy_name = operation.get("strategy", operation_type)
            
            # Map specific strategy to category if needed
            if strategy_name in self.strategy_mapping:
                mapped_strategy = self.strategy_mapping[strategy_name]
                logger.debug(f"Mapped strategy '{strategy_name}' to '{mapped_strategy}'")
            else:
                mapped_strategy = strategy_name
            
            strategy_func = self.cleaning_strategies.get(mapped_strategy)
            
            if not strategy_func:
                # Try to use operation_type as fallback
                strategy_func = self.cleaning_strategies.get(operation_type)
                if not strategy_func:
                    raise ValueError(f"Unknown cleaning strategy: {strategy_name} (mapped to: {mapped_strategy})")
                mapped_strategy = operation_type
            
            # Execute strategy with original strategy name for context
            result_data = strategy_func(data, operation, config)
            
            # Create operation log
            operation_log = {
                "operation_id": operation_id,
                "operation_type": operation_type,
                "strategy": strategy_name,  # Keep original strategy name
                "mapped_strategy": mapped_strategy,  # Add mapped strategy for debugging
                "affected_columns": operation.get("affected_columns", []),
                "parameters": operation.get("parameters", {}),
                "status": "success",
                "timestamp": datetime.now(),
                "rows_affected": len(data) - len(result_data) if len(result_data) != len(data) else 0,
                "description": operation.get("description", "")
            }
            
            return {
                "data": result_data,
                "log": operation_log
            }
            
        except Exception as e:
            logger.error(f"Single operation execution failed: {str(e)}")
            return {
                "data": data,  # Return original data on failure
                "log": {
                    "operation_id": operation_id,
                    "operation_type": operation_type,
                    "status": "failed",
                    "error": str(e),
                    "timestamp": datetime.now()
                }
            }
    
    def _handle_missing_values(self, data: pd.DataFrame, operation: Dict, 
                             config: Dict) -> pd.DataFrame:
        """Handle missing values - cattle data typically doesn't have missing values"""
        # For cattle data, missing values are rare and should be flagged for review
        strategy = operation.get("strategy", "flag_missing")
        affected_columns = operation.get("affected_columns", [])
        
        result_data = data.copy()
        
        if not affected_columns:
            affected_columns = data.columns.tolist()
        
        for col in affected_columns:
            if col not in data.columns:
                continue
                
            if strategy == "flag_missing":
                # Add a flag column for missing values
                missing_mask = result_data[col].isnull()
                if missing_mask.any():
                    result_data[f'{col}_needs_review'] = missing_mask
        
        return result_data
    
    def _treat_outliers(self, data: pd.DataFrame, operation: Dict, 
                       config: Dict) -> pd.DataFrame:
        """Handle outlier treatment for cattle weights"""
        strategy = operation.get("strategy", "remove_outliers")
        affected_lots = operation.get("affected_lots", [])
        
        result_data = data.copy()
        
        if strategy == "remove_outliers":
            # Remove rows with outlier weights
            if affected_lots:
                mask = ~result_data['lot_id'].isin(affected_lots)
                result_data = result_data[mask]
                logger.info(f"Removed {len(affected_lots)} outlier lots")
        
        elif strategy == "manual_review":
            # Add review flag for questionable weights
            if affected_lots:
                mask = result_data['lot_id'].isin(affected_lots)
                result_data.loc[mask, 'needs_manual_review'] = True
                result_data.loc[mask, 'review_reason'] = 'questionable_weights'
                logger.info(f"Flagged {len(affected_lots)} lots for manual review")
        
        return result_data
    
    def _validate_data_integrity(self, data: pd.DataFrame, operation: Dict, 
                               config: Dict) -> pd.DataFrame:
        """Validate and correct data integrity issues"""
        strategy = operation.get("strategy", "validate_constraints")
        affected_lots = operation.get("affected_lots", [])
        parameters = operation.get("parameters", {})
        
        result_data = data.copy()
        
        if strategy == "validate_constraints" and parameters.get("method") == "correct_labels":
            # Correct ready_to_load labels
            target_column = parameters.get("target_column", "ready_to_load")
            new_value = parameters.get("new_value", "Yes")
            
            if affected_lots:
                mask = result_data['lot_id'].isin(affected_lots)
                result_data.loc[mask, target_column] = new_value
                logger.info(f"Corrected {target_column} labels for {len(affected_lots)} lots")
        
        return result_data
    
    def _handle_duplicates(self, data: pd.DataFrame, operation: Dict, 
                         config: Dict) -> pd.DataFrame:
        """Handle duplicate data - implement based on your specific needs"""
        strategy = operation.get("strategy", "remove_duplicates")
        keep = operation.get("parameters", {}).get("keep", "first")
        
        if strategy == "remove_duplicates":
            return data.drop_duplicates(keep=keep)
        
        return data
    
    def _standardize_formats(self, data: pd.DataFrame, operation: Dict, 
                           config: Dict) -> pd.DataFrame:
        """Standardize formats - implement based on your specific needs"""
        affected_columns = operation.get("affected_columns", [])
        format_type = operation.get("parameters", {}).get("format_type", "general")
        
        result_data = data.copy()
        
        for col in affected_columns:
            if col not in data.columns:
                continue
            
            if format_type == "date":
                # Standardize date format
                try:
                    result_data[col] = pd.to_datetime(result_data[col], errors='coerce')
                except:
                    pass
            elif format_type == "case":
                # Standardize case
                case_type = operation.get("parameters", {}).get("case", "lower")
                if case_type == "lower":
                    result_data[col] = result_data[col].astype(str).str.lower()
                elif case_type == "upper":
                    result_data[col] = result_data[col].astype(str).str.upper()
                elif case_type == "title":
                    result_data[col] = result_data[col].astype(str).str.title()
        
        return result_data
    
    def _treat_outliers(self, data: pd.DataFrame, operation: Dict, 
                       config: Dict) -> pd.DataFrame:
        """Treat outliers - implement based on your specific needs"""
        affected_columns = operation.get("affected_columns", [])
        action = operation.get("parameters", {}).get("action", "flag")
        method = operation.get("parameters", {}).get("method", "iqr")
        
        result_data = data.copy()
        
        for col in affected_columns:
            if col not in data.columns or data[col].dtype not in ['int64', 'float64']:
                continue
            
            # Detect outliers using IQR method
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_mask = (data[col] < lower_bound) | (data[col] > upper_bound)
            
            if action == "remove":
                result_data = result_data[~outlier_mask]
            elif action == "flag":
                result_data[f"{col}_outlier_flag"] = outlier_mask
            elif action == "transform":
                # Cap outliers to bounds
                result_data.loc[data[col] < lower_bound, col] = lower_bound
                result_data.loc[data[col] > upper_bound, col] = upper_bound
        
        return result_data
    
    def _validate_data_integrity(self, data: pd.DataFrame, operation: Dict, 
                               config: Dict) -> pd.DataFrame:
        """Validate data integrity - implement based on your specific needs"""
        # This is a placeholder for data validation logic
        # You can implement specific validation rules here
        return data
    
    def _convert_data_types(self, data: pd.DataFrame, operation: Dict, 
                          config: Dict) -> pd.DataFrame:
        """Convert data types - implement based on your specific needs"""
        affected_columns = operation.get("affected_columns", [])
        target_type = operation.get("parameters", {}).get("target_type", "string")
        
        result_data = data.copy()
        
        for col in affected_columns:
            if col not in data.columns:
                continue
            
            try:
                if target_type == "numeric":
                    result_data[col] = pd.to_numeric(result_data[col], errors='coerce')
                elif target_type == "string":
                    result_data[col] = result_data[col].astype(str)
                elif target_type == "datetime":
                    result_data[col] = pd.to_datetime(result_data[col], errors='coerce')
                elif target_type == "boolean":
                    result_data[col] = result_data[col].astype(bool)
            except Exception as e:
                logger.warning(f"Type conversion failed for column {col}: {str(e)}")
        
        return result_data
    
    def _calculate_improvement_metrics(self, original_data: pd.DataFrame, 
                                     cleaned_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate improvement metrics"""
        try:
            metrics = {}
            
            # Completeness improvement
            original_completeness = 1 - (original_data.isnull().sum().sum() / (len(original_data) * len(original_data.columns)))
            cleaned_completeness = 1 - (cleaned_data.isnull().sum().sum() / (len(cleaned_data) * len(cleaned_data.columns)))
            metrics["completeness_improvement"] = cleaned_completeness - original_completeness
            
            # Uniqueness improvement (based on duplicate reduction)
            original_duplicates = original_data.duplicated().sum()
            cleaned_duplicates = cleaned_data.duplicated().sum()
            original_uniqueness = 1 - (original_duplicates / len(original_data))
            cleaned_uniqueness = 1 - (cleaned_duplicates / len(cleaned_data)) if len(cleaned_data) > 0 else 1
            metrics["uniqueness_improvement"] = cleaned_uniqueness - original_uniqueness
            
            # Overall improvement (average of all metrics)
            metrics["overall_improvement"] = np.mean([
                metrics["completeness_improvement"],
                metrics["uniqueness_improvement"]
            ])
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating improvement metrics: {str(e)}")
            return {"overall_improvement": 0.0}
    
    # Helper methods
    def _parse_data_to_dataframe(self, data: str) -> Optional[pd.DataFrame]:
        """Parse data to DataFrame"""
        try:
            # Try CSV format
            from io import StringIO
            return pd.read_csv(StringIO(data))
        except:
            try:
                # Try JSON format
                return pd.read_json(StringIO(data))
            except:
                return None
    
    def _dataframe_to_string(self, df: pd.DataFrame, original_format: str) -> str:
        """Convert DataFrame back to string format"""
        try:
            # Determine original format and convert accordingly
            if "," in original_format and "\n" in original_format:
                # Likely CSV format
                return df.to_csv(index=False)
            else:
                # Default to CSV
                return df.to_csv(index=False)
        except Exception as e:
            logger.error(f"Error converting DataFrame to string: {str(e)}")
            return original_format  # Return original on error

