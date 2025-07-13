"""
Agents package initialization.

This package contains all the AI agents for the data cleaning system.
"""

# Import only the working components
try:
    from .main_controller_fixed import SimpleDataCleaner, process_cleaning_request
    __all__ = ["SimpleDataCleaner", "process_cleaning_request"]
except ImportError as e:
    print(f"Warning: Could not import main controller: {e}")
    __all__ = []

# Version info
__version__ = "2.1.0"
__author__ = "Data Cleaning Agent Team"

