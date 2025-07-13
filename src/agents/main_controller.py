"""
Main Controller Agent Implementation - Simplified and Fixed Version

This module implements the main controller that orchestrates the data cleaning workflow.
"""

import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph import StateGraph, END
from loguru import logger

from ..config.settings import get_settings
from ..schemas.state import DataCleaningState, create_initial_state
from ..utils.json_utils import convert_numpy_types


class MainControllerAgent:
    """
    Main controller that orchestrates the entire data cleaning workflow.
    Simplified version that works reliably.
    """
    
    def __init__(self):
        """Initialize the main controller with all necessary components."""
        self.settings = get_settings()
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=self.settings.openai_model,
            temperature=0.1,
            max_tokens=4000
        )
        
        # Build the workflow graph
        self.workflow = self._build_workflow()
        
        logger.info("MainControllerAgent initialized successfully")
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow for data cleaning."""
        workflow = StateGraph(DataCleaningState)
        
        # Add nodes
        workflow.add_node("load_data", self._load_data)
        workflow.add_node("analyze_data", self._analyze_data)
        workflow.add_node("clean_data", self._clean_data)
        workflow.add_node("generate_report", self._generate_report)
        
        # Add edges
        workflow.set_entry_point("load_data")
        workflow.add_edge("load_data", "analyze_data")
        workflow.add_edge("analyze_data", "clean_data")
        workflow.add_edge("clean_data", "generate_report")
        workflow.add_edge("generate_report", END)
        
        return workflow.compile()
    
    def _load_data(self, state: DataCleaningState) -> Dict[str, Any]:
        """Load and validate the input data."""
        try:
            logger.info(f"Loading data from: {state['data_source']}")
            
            # Read the data file
            data_path = Path(state["data_source"])
            if not data_path.exists():
                raise FileNotFoundError(f"Data file not found: {data_path}")
            
            with open(data_path, 'r', encoding='utf-8') as f:
                raw_data = f.read()
            
            # Determine data format
            if data_path.suffix.lower() == '.csv':
                data_format = 'csv'
            elif data_path.suffix.lower() in ['.json']:
                data_format = 'json'
            else:
                data_format = 'text'
            
            logger.info(f"Data loaded successfully. Format: {data_format}, Size: {len(raw_data)} chars")
            
            return {
                "raw_data": raw_data,
                "data_format": data_format,
                "current_phase": "data_loaded",
                "progress": 25
            }
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return {
                "error": f"Failed to load data: {str(e)}",
                "current_phase": "error",
                "progress": 0
            }
    
    def _analyze_data(self, state: DataCleaningState) -> Dict[str, Any]:
        """Analyze the data and identify issues specific to cattle data."""
        try:
            logger.info("Analyzing cattle data for quality issues...")
            
            raw_data = state["raw_data"]
            
            # Parse CSV data for cattle analysis
            lines = raw_data.strip().split('\n')
            if len(lines) < 2:
                raise ValueError("Invalid CSV format: insufficient data")
            
            headers = [h.strip() for h in lines[0].split(',')]
            data_rows = []
            
            for line in lines[1:]:
                if line.strip():
                    row = [cell.strip() for cell in line.split(',')]
                    if len(row) == len(headers):
                        data_rows.append(dict(zip(headers, row)))
            
            # Cattle-specific analysis
            analysis_result = self._analyze_cattle_data(data_rows)
            
            # Generate cleaning plan
            cleaning_plan = self._generate_cattle_cleaning_plan(analysis_result)
            
            logger.info(f"Analysis completed. Found {len(analysis_result.get('issues', []))} issues")
            
            return {
                "analysis_result": analysis_result,
                "cleaning_plan": cleaning_plan,
                "current_phase": "data_analyzed",
                "progress": 50
            }
            
        except Exception as e:
            logger.error(f"Error analyzing data: {e}")
            return {
                "error": f"Failed to analyze data: {str(e)}",
                "current_phase": "error",
                "progress": 25
            }
    
    def _analyze_cattle_data(self, data_rows: List[Dict[str, str]]) -> Dict[str, Any]:
        """Analyze cattle data for specific issues."""
        issues = []
        outliers = []
        suspicious = []
        label_errors = []
        
        # Define weight thresholds (in pounds)
        normal_entry_weight = (800, 1200)  # Normal range
        normal_exit_weight = (1100, 1600)  # Normal range
        max_threshold_multiplier = 1.3
        
        for row in data_rows:
            lot_id = row.get('lot_id', 'Unknown')
            
            try:
                entry_weight = float(row.get('entry_weight_lbs', 0))
                exit_weight = float(row.get('exit_weight_lbs', 0))
                ready_to_load = row.get('ready_to_load', '').strip().lower()
                
                # Check for extreme outliers (>1.3x threshold)
                if (entry_weight > normal_entry_weight[1] * max_threshold_multiplier or 
                    exit_weight > normal_exit_weight[1] * max_threshold_multiplier or
                    entry_weight < normal_entry_weight[0] * 0.5 or
                    exit_weight < normal_exit_weight[0] * 0.5):
                    
                    outliers.append({
                        'lot_id': lot_id,
                        'entry_weight': entry_weight,
                        'exit_weight': exit_weight,
                        'issue': 'extreme_outlier',
                        'action': 'delete'
                    })
                    issues.append(f"LOT {lot_id}: Extreme weight outlier - DELETE")
                
                # Check for suspicious weights (1.0-1.3x threshold)
                elif (entry_weight > normal_entry_weight[1] or 
                      exit_weight > normal_exit_weight[1] or
                      entry_weight < normal_entry_weight[0] or
                      exit_weight < normal_exit_weight[0]):
                    
                    suspicious.append({
                        'lot_id': lot_id,
                        'entry_weight': entry_weight,
                        'exit_weight': exit_weight,
                        'issue': 'suspicious_weight',
                        'action': 'manual_review'
                    })
                    issues.append(f"LOT {lot_id}: Suspicious weight - MANUAL REVIEW")
                
                # Check for label errors (good weights but marked as not ready)
                elif (normal_entry_weight[0] <= entry_weight <= normal_entry_weight[1] and
                      normal_exit_weight[0] <= exit_weight <= normal_exit_weight[1] and
                      ready_to_load in ['no', 'false', '0']):
                    
                    label_errors.append({
                        'lot_id': lot_id,
                        'entry_weight': entry_weight,
                        'exit_weight': exit_weight,
                        'current_label': ready_to_load,
                        'issue': 'incorrect_label',
                        'action': 'correct_label'
                    })
                    issues.append(f"LOT {lot_id}: Incorrect label - CORRECT TO YES")
                    
            except (ValueError, TypeError) as e:
                issues.append(f"LOT {lot_id}: Invalid weight data - {str(e)}")
        
        return {
            'total_lots': len(data_rows),
            'issues_found': len(issues),
            'outliers': outliers,
            'suspicious': suspicious,
            'label_errors': label_errors,
            'issues': issues,
            'summary': {
                'extreme_outliers': len(outliers),
                'suspicious_weights': len(suspicious),
                'label_corrections': len(label_errors)
            }
        }
    
    def _generate_cattle_cleaning_plan(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate a cleaning plan based on cattle data analysis."""
        plan = []
        
        # Plan for extreme outliers - DELETE
        if analysis['outliers']:
            plan.append({
                'operation': 'delete_outliers',
                'description': f"Delete {len(analysis['outliers'])} lots with extreme weight outliers",
                'affected_lots': [item['lot_id'] for item in analysis['outliers']],
                'priority': 1
            })
        
        # Plan for suspicious weights - MANUAL REVIEW
        if analysis['suspicious']:
            plan.append({
                'operation': 'flag_for_review',
                'description': f"Flag {len(analysis['suspicious'])} lots with suspicious weights for manual review",
                'affected_lots': [item['lot_id'] for item in analysis['suspicious']],
                'priority': 2
            })
        
        # Plan for label corrections - AUTO CORRECT
        if analysis['label_errors']:
            plan.append({
                'operation': 'correct_labels',
                'description': f"Correct ready_to_load labels for {len(analysis['label_errors'])} lots",
                'affected_lots': [item['lot_id'] for item in analysis['label_errors']],
                'priority': 3
            })
        
        return plan
    
    def _clean_data(self, state: DataCleaningState) -> Dict[str, Any]:
        """Execute the cleaning plan on the cattle data."""
        try:
            logger.info("Executing cattle data cleaning plan...")
            
            raw_data = state["raw_data"]
            cleaning_plan = state["cleaning_plan"]
            analysis_result = state["analysis_result"]
            
            # Parse the original data
            lines = raw_data.strip().split('\n')
            headers = [h.strip() for h in lines[0].split(',')]
            data_rows = []
            
            for line in lines[1:]:
                if line.strip():
                    row = [cell.strip() for cell in line.split(',')]
                    if len(row) == len(headers):
                        data_rows.append(dict(zip(headers, row)))
            
            # Execute cleaning operations
            cleaned_rows = []
            operations_log = []
            
            # Get lots to delete and correct
            lots_to_delete = set()
            lots_to_flag = set()
            lots_to_correct = set()
            
            for outlier in analysis_result.get('outliers', []):
                lots_to_delete.add(outlier['lot_id'])
            
            for suspicious in analysis_result.get('suspicious', []):
                lots_to_flag.add(suspicious['lot_id'])
            
            for label_error in analysis_result.get('label_errors', []):
                lots_to_correct.add(label_error['lot_id'])
            
            # Process each row
            for row in data_rows:
                lot_id = row.get('lot_id', '')
                
                if lot_id in lots_to_delete:
                    operations_log.append(f"DELETED: {lot_id} - Extreme weight outlier")
                    continue  # Skip this row (delete it)
                
                elif lot_id in lots_to_flag:
                    row['review_flag'] = 'MANUAL_REVIEW_REQUIRED'
                    operations_log.append(f"FLAGGED: {lot_id} - Suspicious weight for manual review")
                    cleaned_rows.append(row)
                
                elif lot_id in lots_to_correct:
                    row['ready_to_load'] = 'Yes'
                    operations_log.append(f"CORRECTED: {lot_id} - Changed ready_to_load to Yes")
                    cleaned_rows.append(row)
                
                else:
                    cleaned_rows.append(row)  # Keep as is
            
            # Generate cleaned CSV
            if cleaned_rows:
                # Add review_flag column if not exists
                if 'review_flag' not in headers and any('review_flag' in row for row in cleaned_rows):
                    headers.append('review_flag')
                
                cleaned_csv_lines = [','.join(headers)]
                for row in cleaned_rows:
                    csv_row = []
                    for header in headers:
                        csv_row.append(row.get(header, ''))
                    cleaned_csv_lines.append(','.join(csv_row))
                
                cleaned_data = '\n'.join(cleaned_csv_lines)
            else:
                cleaned_data = ','.join(headers) + '\n'  # Empty data with headers
            
            logger.info(f"Cleaning completed. {len(cleaned_rows)} rows remaining from {len(data_rows)} original rows")
            
            return {
                "cleaned_data": cleaned_data,
                "operations_log": operations_log,
                "cleaning_summary": {
                    "original_rows": len(data_rows),
                    "cleaned_rows": len(cleaned_rows),
                    "deleted_rows": len(lots_to_delete),
                    "flagged_rows": len(lots_to_flag),
                    "corrected_rows": len(lots_to_correct)
                },
                "current_phase": "data_cleaned",
                "progress": 75
            }
            
        except Exception as e:
            logger.error(f"Error cleaning data: {e}")
            return {
                "error": f"Failed to clean data: {str(e)}",
                "current_phase": "error",
                "progress": 50
            }
    
    def _generate_report(self, state: DataCleaningState) -> Dict[str, Any]:
        """Generate a comprehensive report of the cleaning process."""
        try:
            logger.info("Generating final report...")
            
            analysis_result = state.get("analysis_result", {})
            cleaning_summary = state.get("cleaning_summary", {})
            operations_log = state.get("operations_log", [])
            
            # Generate report content
            report_sections = []
            
            # Executive Summary
            report_sections.append("# 🐄 Cattle Data Cleaning Report")
            report_sections.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_sections.append("")
            
            # Summary Statistics
            report_sections.append("## 📊 Executive Summary")
            report_sections.append(f"- **Total Lots Processed:** {cleaning_summary.get('original_rows', 0)}")
            report_sections.append(f"- **Lots Remaining:** {cleaning_summary.get('cleaned_rows', 0)}")
            report_sections.append(f"- **Lots Deleted:** {cleaning_summary.get('deleted_rows', 0)} (extreme outliers)")
            report_sections.append(f"- **Lots Flagged:** {cleaning_summary.get('flagged_rows', 0)} (manual review required)")
            report_sections.append(f"- **Labels Corrected:** {cleaning_summary.get('corrected_rows', 0)} (auto-corrected to ready)")
            report_sections.append("")
            
            # Problem Classification
            report_sections.append("## 🔍 Problem Classification")
            
            # Category 1: Extreme Outliers (Deleted)
            outliers = analysis_result.get('outliers', [])
            if outliers:
                report_sections.append("### ❌ Category 1: Extreme Weight Outliers (DELETED)")
                report_sections.append("**Action:** Automatically deleted due to impossible weight values")
                for outlier in outliers:
                    report_sections.append(f"- **{outlier['lot_id']}**: Entry={outlier['entry_weight']}lbs, Exit={outlier['exit_weight']}lbs")
                report_sections.append("")
            
            # Category 2: Suspicious Weights (Manual Review)
            suspicious = analysis_result.get('suspicious', [])
            if suspicious:
                report_sections.append("### ⚠️ Category 2: Suspicious Weights (MANUAL REVIEW REQUIRED)")
                report_sections.append("**Action:** Flagged for human verification - weights outside normal range but potentially valid")
                for susp in suspicious:
                    report_sections.append(f"- **{susp['lot_id']}**: Entry={susp['entry_weight']}lbs, Exit={susp['exit_weight']}lbs")
                report_sections.append("")
            
            # Category 3: Label Corrections (Auto-corrected)
            label_errors = analysis_result.get('label_errors', [])
            if label_errors:
                report_sections.append("### ✅ Category 3: Label Corrections (AUTO-CORRECTED)")
                report_sections.append("**Action:** Automatically corrected ready_to_load status for lots with normal weights")
                for label_error in label_errors:
                    report_sections.append(f"- **{label_error['lot_id']}**: Entry={label_error['entry_weight']}lbs, Exit={label_error['exit_weight']}lbs → Changed to 'Yes'")
                report_sections.append("")
            
            # Next Steps
            report_sections.append("## 🎯 Next Steps Required")
            if suspicious:
                report_sections.append("### Manual Review Required")
                report_sections.append(f"Please review {len(suspicious)} lots flagged with 'MANUAL_REVIEW_REQUIRED' in the cleaned data.")
                report_sections.append("These lots have weights outside normal ranges but may be legitimate.")
                report_sections.append("")
            
            if not outliers and not suspicious and not label_errors:
                report_sections.append("✅ **No issues found!** All cattle data appears to be within normal parameters.")
                report_sections.append("")
            
            # Data Quality Score
            total_issues = len(outliers) + len(suspicious) + len(label_errors)
            total_lots = analysis_result.get('total_lots', 1)
            quality_score = max(0, (total_lots - total_issues) / total_lots * 100)
            
            report_sections.append("## 📈 Data Quality Assessment")
            report_sections.append(f"**Overall Quality Score:** {quality_score:.1f}%")
            
            if quality_score >= 95:
                report_sections.append("🟢 **Excellent** - Minimal data quality issues")
            elif quality_score >= 85:
                report_sections.append("🟡 **Good** - Some issues identified and resolved")
            elif quality_score >= 70:
                report_sections.append("🟠 **Fair** - Multiple issues requiring attention")
            else:
                report_sections.append("🔴 **Poor** - Significant data quality problems")
            
            final_report = '\n'.join(report_sections)
            
            logger.info("Report generation completed successfully")
            
            return {
                "final_report": final_report,
                "quality_score": quality_score,
                "current_phase": "completed",
                "progress": 100,
                "status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return {
                "error": f"Failed to generate report: {str(e)}",
                "current_phase": "error",
                "progress": 75
            }
    
    async def process_data(self, user_requirements: str, data_source: str) -> Dict[str, Any]:
        """Process data cleaning request using the workflow."""
        try:
            session_id = str(uuid.uuid4())
            start_time = datetime.now()
            
            logger.info(f"Starting data cleaning process. Session: {session_id}")
            
            # Create initial state
            initial_state = create_initial_state(
                session_id=session_id,
                user_requirements=user_requirements,
                data_source=data_source
            )
            
            # Run the workflow
            result = await self.workflow.ainvoke(initial_state)
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Convert numpy types for JSON serialization
            safe_result = convert_numpy_types(result)
            safe_result["execution_time"] = execution_time
            
            logger.info(f"Data cleaning process completed in {execution_time:.2f}s")
            
            return safe_result
            
        except Exception as e:
            logger.error(f"Error in data cleaning process: {e}")
            return {
                "status": "error",
                "error": str(e),
                "current_phase": "error",
                "progress": 0
            }


# Convenience function for external use
async def process_cleaning_request(user_requirements: str, data_source: str) -> Dict[str, Any]:
    """
    Process a data cleaning request.
    
    Args:
        user_requirements: Natural language description of cleaning requirements
        data_source: Path to the data file to be cleaned
        
    Returns:
        Dictionary containing the cleaning results and report
    """
    controller = MainControllerAgent()
    return await controller.process_data(user_requirements, data_source)

