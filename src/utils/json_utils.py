"""
JSON Utility Functions

This module provides utility functions for handling JSON serialization
of numpy and pandas data types that are not natively JSON serializable.
"""

import json
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Union


def convert_numpy_types(obj: Any) -> Any:
    """
    Convert numpy and pandas types to Python native types for JSON serialization.
    
    Args:
        obj: Object that may contain numpy/pandas types
        
    Returns:
        Object with numpy/pandas types converted to Python native types
    """
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif pd.isna(obj):
        return None
    else:
        return obj


def safe_json_dumps(obj: Any, **kwargs) -> str:
    """
    Safely serialize object to JSON string, handling numpy/pandas types.
    
    Args:
        obj: Object to serialize
        **kwargs: Additional arguments for json.dumps
        
    Returns:
        JSON string
    """
    converted_obj = convert_numpy_types(obj)
    return json.dumps(converted_obj, **kwargs)


def safe_json_loads(json_str: str, **kwargs) -> Any:
    """
    Safely deserialize JSON string to Python object.
    
    Args:
        json_str: JSON string to deserialize
        **kwargs: Additional arguments for json.loads
        
    Returns:
        Python object
    """
    return json.loads(json_str, **kwargs)


def prepare_for_llm(data: Any) -> Any:
    """
    Prepare data for LLM consumption by ensuring all types are JSON serializable.
    
    Args:
        data: Data to prepare
        
    Returns:
        Data with all types converted to JSON serializable formats
    """
    return convert_numpy_types(data)


class NumpyJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that handles numpy and pandas types.
    """
    
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif pd.isna(obj):
            return None
        
        return super().default(obj)


def validate_json_serializable(obj: Any) -> bool:
    """
    Check if an object is JSON serializable.
    
    Args:
        obj: Object to check
        
    Returns:
        True if serializable, False otherwise
    """
    try:
        json.dumps(obj, cls=NumpyJSONEncoder)
        return True
    except (TypeError, ValueError):
        return False


def clean_for_json(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean a dictionary to ensure all values are JSON serializable.
    
    Args:
        data: Dictionary to clean
        
    Returns:
        Cleaned dictionary
    """
    cleaned = {}
    
    for key, value in data.items():
        try:
            # Test if value is JSON serializable
            json.dumps(value, cls=NumpyJSONEncoder)
            cleaned[key] = convert_numpy_types(value)
        except (TypeError, ValueError):
            # If not serializable, convert to string representation
            cleaned[key] = str(value)
    
    return cleaned

