"""
Main Controller Agent Implementation - Fixed Version

This module implements a simplified, working version of the main controller.
"""

import asyncio
import json
import uuid
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from ..config.settings import get_settings
from ..utils.json_utils import convert_numpy_types


class SimpleDataCleaner:
    """
    Simplified data cleaner that works reliably for cattle data.
    """
    
    def __init__(self):
        """Initialize the data cleaner."""
        self.settings = get_settings()
        self.llm = ChatOpenAI(
            model=self.settings.llm.model,
            temperature=0.1,
            max_tokens=2000
        )
        logger.info("SimpleDataCleaner initialized successfully")
    
    def process_cattle_data(self, data_path: str, requirements: str) -> Dict[str, Any]:
        """
        Process cattle data with simplified logic.
        
        Args:
            data_path: Path to the CSV file
            requirements: User requirements
            
        Returns:
            Dict containing processing results
        """
        try:
            start_time = datetime.now()
            
            # Load data
            logger.info(f"Loading data from {data_path}")
            df = pd.read_csv(data_path)
            
            # Analyze data
            analysis_result = self._analyze_cattle_data(df)
            
            # Clean data
            cleaned_df, cleaning_log = self._clean_cattle_data(df, analysis_result)
            
            # Generate report
            report = self._generate_report(df, cleaned_df, analysis_result, cleaning_log)
            
            # Calculate metrics
            execution_time = (datetime.now() - start_time).total_seconds()
            quality_score = self._calculate_quality_score(df, cleaned_df)
            
            # Save cleaned data
            cleaned_data_csv = cleaned_df.to_csv(index=False)
            
            return {
                "status": "completed",
                "cleaned_data": cleaned_data_csv,
                "final_report": report,
                "execution_time": execution_time,
                "quality_score": quality_score,
                "original_rows": len(df),
                "cleaned_rows": len(cleaned_df),
                "analysis_result": analysis_result,
                "cleaning_log": cleaning_log
            }
            
        except Exception as e:
            logger.error(f"Error processing cattle data: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "execution_time": 0.0,
                "quality_score": 0.0
            }
    
    def _analyze_cattle_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze cattle data to identify issues."""
        try:
            analysis = {
                "total_lots": len(df),
                "extreme_outliers": [],
                "suspicious_weights": [],
                "label_errors": [],
                "weight_thresholds": {
                    "min_normal": 500,   # lbs
                    "max_normal": 1500,  # lbs  
                    "extreme_threshold": 1950  # 1.3 * 1500
                }
            }
            
            # Check for extreme outliers (>1.3x threshold)
            for idx, row in df.iterrows():
                entry_weight = row.get('entry_weight_lbs', 0)
                exit_weight = row.get('exit_weight_lbs', 0)
                
                if entry_weight > analysis["weight_thresholds"]["extreme_threshold"] or \
                   exit_weight > analysis["weight_thresholds"]["extreme_threshold"]:
                    analysis["extreme_outliers"].append({
                        "lot_id": row.get('lot_id', f'Row_{idx}'),
                        "entry_weight": entry_weight,
                        "exit_weight": exit_weight,
                        "reason": "Weight exceeds 1.3x normal threshold"
                    })
                
                # Check for suspicious weights (1.0-1.3x threshold)
                elif entry_weight > analysis["weight_thresholds"]["max_normal"] or \
                     exit_weight > analysis["weight_thresholds"]["max_normal"]:
                    analysis["suspicious_weights"].append({
                        "lot_id": row.get('lot_id', f'Row_{idx}'),
                        "entry_weight": entry_weight,
                        "exit_weight": exit_weight,
                        "reason": "Weight in suspicious range (1.0-1.3x threshold)"
                    })
                
                # Check for label errors (normal weight but marked as not ready)
                ready_to_load = str(row.get('ready_to_load', '')).lower()
                if (entry_weight <= analysis["weight_thresholds"]["max_normal"] and 
                    exit_weight <= analysis["weight_thresholds"]["max_normal"] and
                    entry_weight >= analysis["weight_thresholds"]["min_normal"] and
                    exit_weight >= analysis["weight_thresholds"]["min_normal"] and
                    ready_to_load in ['no', 'false', '0', 'n']):
                    analysis["label_errors"].append({
                        "lot_id": row.get('lot_id', f'Row_{idx}'),
                        "entry_weight": entry_weight,
                        "exit_weight": exit_weight,
                        "current_label": ready_to_load,
                        "reason": "Normal weight but marked as not ready"
                    })
            
            logger.info(f"Analysis completed: {len(analysis['extreme_outliers'])} extreme outliers, "
                       f"{len(analysis['suspicious_weights'])} suspicious weights, "
                       f"{len(analysis['label_errors'])} label errors")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in data analysis: {str(e)}")
            return {"error": str(e)}
    
    def _clean_cattle_data(self, df: pd.DataFrame, analysis: Dict[str, Any]) -> tuple:
        """Clean the cattle data based on analysis results."""
        try:
            cleaned_df = df.copy()
            cleaning_log = []
            
            # 1. Remove extreme outliers
            extreme_lot_ids = [item["lot_id"] for item in analysis.get("extreme_outliers", [])]
            if extreme_lot_ids:
                cleaned_df = cleaned_df[~cleaned_df['lot_id'].isin(extreme_lot_ids)]
                cleaning_log.append({
                    "action": "remove_extreme_outliers",
                    "affected_lots": extreme_lot_ids,
                    "count": len(extreme_lot_ids),
                    "reason": "Weight exceeds 1.3x normal threshold"
                })
            
            # 2. Mark suspicious weights for manual review
            suspicious_lot_ids = [item["lot_id"] for item in analysis.get("suspicious_weights", [])]
            if suspicious_lot_ids:
                cleaned_df.loc[cleaned_df['lot_id'].isin(suspicious_lot_ids), 'manual_review_required'] = 'Yes'
                cleaning_log.append({
                    "action": "mark_for_manual_review",
                    "affected_lots": suspicious_lot_ids,
                    "count": len(suspicious_lot_ids),
                    "reason": "Weight in suspicious range (1.0-1.3x threshold)"
                })
            
            # 3. Fix label errors
            label_error_lot_ids = [item["lot_id"] for item in analysis.get("label_errors", [])]
            if label_error_lot_ids:
                cleaned_df.loc[cleaned_df['lot_id'].isin(label_error_lot_ids), 'ready_to_load'] = 'Yes'
                cleaning_log.append({
                    "action": "fix_label_errors",
                    "affected_lots": label_error_lot_ids,
                    "count": len(label_error_lot_ids),
                    "reason": "Normal weight but incorrectly marked as not ready"
                })
            
            # Add manual review column if not exists
            if 'manual_review_required' not in cleaned_df.columns:
                cleaned_df['manual_review_required'] = 'No'
            
            logger.info(f"Cleaning completed: {len(df)} -> {len(cleaned_df)} rows")
            
            return cleaned_df, cleaning_log
            
        except Exception as e:
            logger.error(f"Error in data cleaning: {str(e)}")
            return df, [{"error": str(e)}]
    
    def _generate_report(self, original_df: pd.DataFrame, cleaned_df: pd.DataFrame, 
                        analysis: Dict[str, Any], cleaning_log: List[Dict]) -> str:
        """Generate a comprehensive cleaning report."""
        try:
            report_lines = []
            report_lines.append("🐄 CATTLE DATA CLEANING REPORT")
            report_lines.append("=" * 50)
            report_lines.append(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_lines.append("")
            
            # Executive Summary
            report_lines.append("📊 EXECUTIVE SUMMARY")
            report_lines.append("-" * 20)
            report_lines.append(f"• Original lots: {len(original_df)}")
            report_lines.append(f"• Processed lots: {len(cleaned_df)}")
            report_lines.append(f"• Removed lots: {len(original_df) - len(cleaned_df)}")
            report_lines.append(f"• Quality improvement: {self._calculate_quality_score(original_df, cleaned_df):.1f}%")
            report_lines.append("")
            
            # Problem Classification
            report_lines.append("🔍 PROBLEM CLASSIFICATION")
            report_lines.append("-" * 25)
            
            # Extreme outliers
            extreme_outliers = analysis.get("extreme_outliers", [])
            report_lines.append(f"1. EXTREME OUTLIERS (Deleted): {len(extreme_outliers)} lots")
            if extreme_outliers:
                for outlier in extreme_outliers:
                    report_lines.append(f"   • {outlier['lot_id']}: Entry={outlier['entry_weight']}kg, Exit={outlier['exit_weight']}kg")
            else:
                report_lines.append("   • No extreme outliers found")
            report_lines.append("")
            
            # Suspicious weights
            suspicious_weights = analysis.get("suspicious_weights", [])
            report_lines.append(f"2. SUSPICIOUS WEIGHTS (Manual Review): {len(suspicious_weights)} lots")
            if suspicious_weights:
                for suspicious in suspicious_weights:
                    report_lines.append(f"   • {suspicious['lot_id']}: Entry={suspicious['entry_weight']}kg, Exit={suspicious['exit_weight']}kg")
            else:
                report_lines.append("   • No suspicious weights found")
            report_lines.append("")
            
            # Label errors
            label_errors = analysis.get("label_errors", [])
            report_lines.append(f"3. LABEL ERRORS (Auto-corrected): {len(label_errors)} lots")
            if label_errors:
                for error in label_errors:
                    report_lines.append(f"   • {error['lot_id']}: Changed from '{error['current_label']}' to 'Yes'")
            else:
                report_lines.append("   • No label errors found")
            report_lines.append("")
            
            # Processing Actions
            report_lines.append("⚙️ PROCESSING ACTIONS")
            report_lines.append("-" * 20)
            for log_entry in cleaning_log:
                action = log_entry.get("action", "unknown")
                count = log_entry.get("count", 0)
                reason = log_entry.get("reason", "No reason provided")
                
                if action == "remove_extreme_outliers":
                    report_lines.append(f"✂️ DELETED {count} lots: {reason}")
                elif action == "mark_for_manual_review":
                    report_lines.append(f"⚠️ FLAGGED {count} lots for manual review: {reason}")
                elif action == "fix_label_errors":
                    report_lines.append(f"🔧 CORRECTED {count} labels: {reason}")
            report_lines.append("")
            
            # Next Steps
            report_lines.append("🎯 NEXT STEPS")
            report_lines.append("-" * 12)
            manual_review_count = len(suspicious_weights)
            if manual_review_count > 0:
                report_lines.append(f"• Review {manual_review_count} lots marked for manual inspection")
                report_lines.append("• Verify weight measurements for flagged lots")
                report_lines.append("• Update ready_to_load status after manual review")
            else:
                report_lines.append("• No manual review required")
                report_lines.append("• All remaining lots are ready for processing")
            report_lines.append("")
            
            # Data Quality Metrics
            report_lines.append("📈 DATA QUALITY METRICS")
            report_lines.append("-" * 23)
            if len(cleaned_df) > 0:
                ready_count = len(cleaned_df[cleaned_df['ready_to_load'].str.lower().isin(['yes', 'true', '1', 'y'])])
                report_lines.append(f"• Ready to load: {ready_count}/{len(cleaned_df)} lots ({ready_count/len(cleaned_df)*100:.1f}%)")
                report_lines.append(f"• Data completeness: {(1 - cleaned_df.isnull().sum().sum() / (len(cleaned_df) * len(cleaned_df.columns))) * 100:.1f}%")
                report_lines.append(f"• Weight range compliance: {((cleaned_df['entry_weight_lbs'] <= 1500) & (cleaned_df['exit_weight_lbs'] <= 1500)).sum()}/{len(cleaned_df)} lots")
            else:
                report_lines.append("• No data remaining after cleaning")
                report_lines.append("• All lots were removed due to extreme outliers")
            
            return "\n".join(report_lines)
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            return f"Report generation failed: {str(e)}"
    
    def _calculate_quality_score(self, original_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> float:
        """Calculate a quality improvement score."""
        try:
            # Simple quality score based on data improvements
            original_issues = 0
            cleaned_issues = 0
            
            # Count weight issues
            for df, counter in [(original_df, "original"), (cleaned_df, "cleaned")]:
                issues = 0
                for _, row in df.iterrows():
                    entry_weight = row.get('entry_weight_lbs', 0)
                    exit_weight = row.get('exit_weight_lbs', 0)
                    ready_to_load = str(row.get('ready_to_load', '')).lower()
                    
                    # Count extreme weights
                    if entry_weight > 1950 or exit_weight > 1950:
                        issues += 1
                    
                    # Count label inconsistencies
                    if (500 <= entry_weight <= 1500 and 500 <= exit_weight <= 1500 and 
                        ready_to_load in ['no', 'false', '0', 'n']):
                        issues += 1
                
                if counter == "original":
                    original_issues = issues
                else:
                    cleaned_issues = issues
            
            # Calculate improvement percentage
            if original_issues == 0:
                return 100.0
            
            improvement = ((original_issues - cleaned_issues) / original_issues) * 100
            return max(0, min(100, improvement))
            
        except Exception as e:
            logger.error(f"Error calculating quality score: {str(e)}")
            return 0.0


async def process_cleaning_request(user_requirements: str, data_source: str) -> Dict[str, Any]:
    """
    Process a data cleaning request using the simplified cleaner.
    
    Args:
        user_requirements: User's cleaning requirements
        data_source: Path to the data file
        
    Returns:
        Dict containing the processing results
    """
    try:
        logger.info(f"Processing cleaning request: {user_requirements}")
        logger.info(f"Data source: {data_source}")
        
        # Initialize the cleaner
        cleaner = SimpleDataCleaner()
        
        # Process the data
        result = cleaner.process_cattle_data(data_source, user_requirements)
        
        # Convert numpy types for JSON serialization
        safe_result = convert_numpy_types(result)
        
        logger.info(f"Processing completed with status: {safe_result.get('status')}")
        
        return safe_result
        
    except Exception as e:
        logger.error(f"Error in process_cleaning_request: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "execution_time": 0.0,
            "quality_score": 0.0
        }

