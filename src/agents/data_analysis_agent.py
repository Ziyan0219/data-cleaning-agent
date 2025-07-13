"""
Data Analysis Agent Implementation

This module implements the data analysis agent responsible for comprehensive
data quality analysis, including statistical analysis, pattern recognition,
and quality issue detection.
"""

import json
import re
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd
import numpy as np
from loguru import logger

from langchain_core.messages import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate

from ..config.settings import get_settings
from ..utils.json_utils import prepare_for_llm, safe_json_dumps


class DataAnalysisAgent:
    """Data Analysis Agent Class"""
    
    def __init__(self, llm):
        self.llm = llm
        self.settings = get_settings()
        self.analysis_prompt = self._create_analysis_prompt()
        
        logger.info("Data Analysis Agent initialized successfully")
    
    def _create_analysis_prompt(self) -> ChatPromptTemplate:
        """Create analysis prompt template"""
        template = self.settings.get_prompt_templates()["data_analysis"]
        
        return ChatPromptTemplate.from_messages([
            ("system", """You are a professional data quality analysis expert. Your tasks are:
1. Analyze data samples and identify various quality issues
2. Assess the severity and impact scope of issues
3. Provide specific cleaning strategy recommendations
4. Estimate cleaning complexity and expected effects

Please always return structured analysis results in JSON format."""),
            ("human", template)
        ])
    
    def analyze_data(self, data: str, user_requirements: str) -> Dict[str, Any]:
        """Analyze data quality with cattle-specific business rules"""
        try:
            logger.info("Starting cattle data quality analysis")
            
            # Parse data
            df = self._parse_data(data)
            
            # Cattle-specific analysis
            cattle_analysis = self._analyze_cattle_data(df)
            
            # Generate schema information
            schema_info = self._generate_schema_info(df)
            
            # Combine all analysis results
            result = {
                "data_overview": {
                    "total_lots": len(df),
                    "columns": list(df.columns),
                    "data_types": {col: str(df[col].dtype) for col in df.columns}
                },
                "cattle_analysis": cattle_analysis,
                "schema_info": schema_info,
                "cleaning_recommendations": self._generate_cattle_cleaning_plan(cattle_analysis),
                "analysis_timestamp": datetime.now().isoformat()
            }
            
            logger.info("Cattle data analysis completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Data analysis failed: {str(e)}")
            return {
                "error": str(e),
                "analysis_timestamp": datetime.now().isoformat()
            }
    
    def _analyze_cattle_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze cattle data for specific business issues"""
        analysis = {
            "weight_analysis": {},
            "outlier_lots": [],
            "review_lots": [],
            "label_issues": [],
            "summary": {}
        }
        
        # Define weight thresholds
        NORMAL_ENTRY_MIN, NORMAL_ENTRY_MAX = 550, 800
        NORMAL_EXIT_MIN, NORMAL_EXIT_MAX = 1100, 1500
        
        OUTLIER_ENTRY_THRESHOLD = NORMAL_ENTRY_MAX * 1.3  # 1040 lbs
        OUTLIER_EXIT_THRESHOLD = NORMAL_EXIT_MAX * 1.3    # 1950 lbs
        
        # Analyze each lot
        for idx, row in df.iterrows():
            lot_id = row.get('lot_id', f'LOT_{idx}')
            entry_weight = row.get('entry_weight_lbs', 0)
            exit_weight = row.get('exit_weight_lbs', 0)
            ready_to_load = str(row.get('ready_to_load', '')).strip().lower()
            
            # Check for outlier weights (>1.3x threshold)
            if (entry_weight > OUTLIER_ENTRY_THRESHOLD or 
                exit_weight > OUTLIER_EXIT_THRESHOLD):
                analysis["outlier_lots"].append({
                    "lot_id": lot_id,
                    "entry_weight": entry_weight,
                    "exit_weight": exit_weight,
                    "issue": "Extreme outlier weights",
                    "action": "DELETE"
                })
            
            # Check for questionable weights (1.0-1.3x threshold)
            elif ((entry_weight > NORMAL_ENTRY_MAX and entry_weight <= OUTLIER_ENTRY_THRESHOLD) or
                  (exit_weight > NORMAL_EXIT_MAX and exit_weight <= OUTLIER_EXIT_THRESHOLD)):
                analysis["review_lots"].append({
                    "lot_id": lot_id,
                    "entry_weight": entry_weight,
                    "exit_weight": exit_weight,
                    "issue": "Questionable weights need review",
                    "action": "MANUAL_REVIEW"
                })
            
            # Check for incorrect ready_to_load labels
            # If weights are in normal range but marked as not ready
            elif (NORMAL_ENTRY_MIN <= entry_weight <= NORMAL_ENTRY_MAX and
                  NORMAL_EXIT_MIN <= exit_weight <= NORMAL_EXIT_MAX and
                  ready_to_load in ['no', 'false', '0']):
                analysis["label_issues"].append({
                    "lot_id": lot_id,
                    "entry_weight": entry_weight,
                    "exit_weight": exit_weight,
                    "current_label": ready_to_load,
                    "issue": "Should be ready to load",
                    "action": "CORRECT_LABEL"
                })
        
        # Generate summary
        analysis["summary"] = {
            "total_lots": len(df),
            "outlier_lots_count": len(analysis["outlier_lots"]),
            "review_lots_count": len(analysis["review_lots"]),
            "label_issues_count": len(analysis["label_issues"]),
            "clean_lots_count": (len(df) - len(analysis["outlier_lots"]) - 
                               len(analysis["review_lots"]) - len(analysis["label_issues"]))
        }
        
        return analysis
    
    def _generate_cattle_cleaning_plan(self, cattle_analysis: Dict) -> List[Dict]:
        """Generate specific cleaning plan for cattle data"""
        cleaning_plan = []
        
        # Plan for outlier deletion
        if cattle_analysis["outlier_lots"]:
            cleaning_plan.append({
                "operation_id": "delete_outliers",
                "operation_type": "outlier_treatment",
                "strategy": "remove_outliers",
                "description": f"Delete {len(cattle_analysis['outlier_lots'])} lots with extreme outlier weights",
                "affected_lots": [lot["lot_id"] for lot in cattle_analysis["outlier_lots"]],
                "parameters": {
                    "method": "delete_rows",
                    "threshold_multiplier": 1.3
                },
                "priority": 1
            })
        
        # Plan for manual review flagging
        if cattle_analysis["review_lots"]:
            cleaning_plan.append({
                "operation_id": "flag_for_review",
                "operation_type": "data_validation",
                "strategy": "manual_review",
                "description": f"Flag {len(cattle_analysis['review_lots'])} lots for manual weight review",
                "affected_lots": [lot["lot_id"] for lot in cattle_analysis["review_lots"]],
                "parameters": {
                    "method": "add_review_flag",
                    "review_reason": "questionable_weights"
                },
                "priority": 2
            })
        
        # Plan for label correction
        if cattle_analysis["label_issues"]:
            cleaning_plan.append({
                "operation_id": "correct_labels",
                "operation_type": "data_validation",
                "strategy": "validate_constraints",
                "description": f"Correct ready_to_load labels for {len(cattle_analysis['label_issues'])} lots",
                "affected_lots": [lot["lot_id"] for lot in cattle_analysis["label_issues"]],
                "parameters": {
                    "method": "correct_labels",
                    "target_column": "ready_to_load",
                    "new_value": "Yes"
                },
                "priority": 3
            })
        
        return cleaning_planDict[str, Any]:
        """Calculate basic statistical information"""
        try:
            # Try to parse as DataFrame
            df = self._parse_data_to_dataframe(data)
            
            if df is None:
                return {"error": "Unable to parse data format"}
            
            stats = {
                "row_count": len(df),
                "column_count": len(df.columns),
                "columns": list(df.columns),
                "data_types": df.dtypes.astype(str).to_dict(),
                "missing_values": df.isnull().sum().to_dict(),
                "missing_percentage": (df.isnull().sum() / len(df) * 100).to_dict(),
                "duplicate_rows": df.duplicated().sum(),
                "memory_usage": df.memory_usage(deep=True).sum()
            }
            
            # Numeric column statistics
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 0:
                stats["numeric_statistics"] = df[numeric_columns].describe().to_dict()
            
            # Text column statistics
            text_columns = df.select_dtypes(include=['object']).columns
            if len(text_columns) > 0:
                text_stats = {}
                for col in text_columns:
                    text_stats[col] = {
                        "unique_values": df[col].nunique(),
                        "most_common": df[col].value_counts().head(5).to_dict()
                    }
                stats["text_statistics"] = text_stats
            
            return stats
            
        except Exception as e:
            logger.error(f"Basic statistics calculation failed: {str(e)}")
            return {"error": str(e)}
    
    def _infer_data_schema(self, data: str) -> Dict[str, Any]:
        """Infer data schema"""
        try:
            df = self._parse_data_to_dataframe(data)
            
            if df is None:
                return {"error": "Unable to infer data schema"}
            
            schema = {
                "format": "tabular",
                "columns": {},
                "constraints": {},
                "relationships": []
            }
            
            for col in df.columns:
                col_data = df[col].dropna()
                
                # Infer column type and characteristics
                column_info = {
                    "data_type": str(df[col].dtype),
                    "nullable": bool(df[col].isnull().any()),  # Convert numpy.bool_ to Python bool
                    "unique_count": int(df[col].nunique()),   # Convert numpy.int64 to Python int
                    "sample_values": col_data.head(5).tolist()
                }
                
                # Special pattern detection
                if col_data.dtype == 'object':
                    column_info.update(self._detect_text_patterns(col_data))
                elif np.issubdtype(col_data.dtype, np.number):
                    column_info.update(self._detect_numeric_patterns(col_data))
                
                schema["columns"][col] = column_info
            
            return schema
            
        except Exception as e:
            logger.error(f"Data schema inference failed: {str(e)}")
            return {"error": str(e)}
    
    def _detect_text_patterns(self, series: pd.Series) -> Dict[str, Any]:
        """Detect text patterns"""
        patterns = {
            "email": r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            "phone": r'^[\+]?[1-9][\d]{0,15}$',
            "url": r'^https?://[^\s/$.?#].[^\s]*$',
            "date": r'^\d{4}-\d{2}-\d{2}$|^\d{2}/\d{2}/\d{4}$',
            "postal_code": r'^\d{5}(-\d{4})?$'
        }
        
        detected_patterns = {}
        sample_values = series.dropna().astype(str)
        
        for pattern_name, pattern in patterns.items():
            matches = sample_values.str.match(pattern, na=False).sum()
            if matches > 0:
                detected_patterns[pattern_name] = {
                    "match_count": matches,
                    "match_percentage": matches / len(sample_values) * 100
                }
        
        # Detect other characteristics
        detected_patterns.update({
            "avg_length": float(sample_values.str.len().mean()),
            "max_length": int(sample_values.str.len().max()),
            "min_length": int(sample_values.str.len().min()),
            "contains_special_chars": bool(sample_values.str.contains(r'[^a-zA-Z0-9\s]').any())
        })
        
        return {"patterns": detected_patterns}
    
    def _detect_numeric_patterns(self, series: pd.Series) -> Dict[str, Any]:
        """Detect numeric patterns"""
        clean_data = series.dropna()
        
        patterns = {
            "range": {"min": clean_data.min(), "max": clean_data.max()},
            "mean": clean_data.mean(),
            "median": clean_data.median(),
            "std": clean_data.std(),
            "outliers": self._detect_outliers(clean_data),
            "distribution": self._analyze_distribution(clean_data)
        }
        
        return {"patterns": patterns}
    
    def _detect_outliers(self, series: pd.Series) -> Dict[str, Any]:
        """Detect outliers"""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = series[(series < lower_bound) | (series > upper_bound)]
        
        return {
            "count": len(outliers),
            "percentage": len(outliers) / len(series) * 100,
            "values": outliers.tolist()[:10]  # Return only first 10 outliers
        }
    
    def _analyze_distribution(self, series: pd.Series) -> str:
        """Analyze data distribution"""
        # Simple distribution analysis
        skewness = series.skew()
        
        if abs(skewness) < 0.5:
            return "normal"
        elif skewness > 0.5:
            return "right_skewed"
        else:
            return "left_skewed"
    
    def _llm_quality_analysis(self, data: str, schema_info: Dict, 
                            user_requirements: str) -> Dict[str, Any]:
        """Use LLM for quality analysis"""
        try:
            # Prepare data sample (limit length)
            data_sample = data[:2000] if len(data) > 2000 else data
            
            # Prepare schema_info for LLM (ensure JSON serializable)
            safe_schema_info = prepare_for_llm(schema_info)
            
            # Build prompt
            prompt = self.analysis_prompt.format_messages(
                data_sample=data_sample,
                schema_info=safe_json_dumps(safe_schema_info, indent=2),
                user_requirements=user_requirements
            )
            
            # Call LLM
            response = self.llm.invoke(prompt)
            
            # Parse response
            return self._parse_llm_response(response.content)
            
        except Exception as e:
            logger.error(f"LLM quality analysis failed: {str(e)}")
            return {"error": str(e)}
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response"""
        try:
            # Try to parse JSON directly
            return json.loads(response)
        except json.JSONDecodeError:
            # If not valid JSON, try to extract JSON part
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
            
            # If unable to parse, return text response
            return {
                "quality_issues": [],
                "severity_assessment": "unknown",
                "cleaning_recommendations": [],
                "raw_response": response
            }
    
    def _rule_based_detection(self, data: str, schema_info: Dict) -> List[Dict]:
        """Rule-based issue detection"""
        issues = []
        
        try:
            df = self._parse_data_to_dataframe(data)
            if df is None:
                return issues
            
            # Detect missing values
            missing_issues = self._detect_missing_values(df)
            issues.extend(missing_issues)
            
            # Detect duplicate data
            duplicate_issues = self._detect_duplicates(df)
            issues.extend(duplicate_issues)
            
            # Detect data type inconsistencies
            type_issues = self._detect_type_inconsistencies(df)
            issues.extend(type_issues)
            
            # Detect format issues
            format_issues = self._detect_format_issues(df)
            issues.extend(format_issues)
            
            # Detect outliers
            outlier_issues = self._detect_outlier_issues(df)
            issues.extend(outlier_issues)
            
        except Exception as e:
            logger.error(f"Rule detection failed: {str(e)}")
            issues.append({
                "type": "detection_error",
                "description": f"Error occurred during rule detection: {str(e)}",
                "severity": "low",
                "affected_columns": []
            })
        
        return issues
    
    def _detect_missing_values(self, df: pd.DataFrame) -> List[Dict]:
        """Detect missing value issues"""
        issues = []
        missing_threshold = 0.1  # 10% missing value threshold
        
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_percentage = missing_count / len(df)
            
            if missing_percentage > missing_threshold:
                issues.append({
                    "type": "missing_values",
                    "description": f"Column '{col}' has {missing_percentage:.1%} missing values",
                    "severity": "high" if missing_percentage > 0.5 else "medium",
                    "affected_columns": [col],
                    "details": {
                        "missing_count": missing_count,
                        "missing_percentage": missing_percentage
                    }
                })
        
        return issues
    
    def _detect_duplicates(self, df: pd.DataFrame) -> List[Dict]:
        """Detect duplicate data issues"""
        issues = []
        duplicate_count = df.duplicated().sum()
        
        if duplicate_count > 0:
            duplicate_percentage = duplicate_count / len(df)
            issues.append({
                "type": "duplicate_rows",
                "description": f"Found {duplicate_count} duplicate rows ({duplicate_percentage:.1%})",
                "severity": "medium" if duplicate_percentage < 0.1 else "high",
                "affected_columns": list(df.columns),
                "details": {
                    "duplicate_count": duplicate_count,
                    "duplicate_percentage": duplicate_percentage
                }
            })
        
        return issues
    
    def _detect_type_inconsistencies(self, df: pd.DataFrame) -> List[Dict]:
        """Detect data type inconsistency issues"""
        issues = []
        
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if should be numeric type
                non_null_values = df[col].dropna()
                numeric_count = 0
                
                for value in non_null_values:
                    try:
                        float(str(value))
                        numeric_count += 1
                    except ValueError:
                        pass
                
                if numeric_count > len(non_null_values) * 0.8:  # 80% are numeric
                    issues.append({
                        "type": "type_inconsistency",
                        "description": f"Column '{col}' might should be numeric type",
                        "severity": "medium",
                        "affected_columns": [col],
                        "details": {
                            "current_type": "object",
                            "suggested_type": "numeric",
                            "numeric_percentage": numeric_count / len(non_null_values)
                        }
                    })
        
        return issues
    
    def _detect_format_issues(self, df: pd.DataFrame) -> List[Dict]:
        """Detect format issues"""
        issues = []
        
        for col in df.columns:
            if df[col].dtype == 'object':
                # Detect inconsistent date formats
                date_formats = self._detect_date_formats(df[col])
                if len(date_formats) > 1:
                    issues.append({
                        "type": "inconsistent_date_format",
                        "description": f"Column '{col}' has multiple date formats",
                        "severity": "medium",
                        "affected_columns": [col],
                        "details": {"detected_formats": date_formats}
                    })
                
                # Detect case inconsistency
                if self._has_case_inconsistency(df[col]):
                    issues.append({
                        "type": "case_inconsistency",
                        "description": f"Column '{col}' has case inconsistency",
                        "severity": "low",
                        "affected_columns": [col]
                    })
        
        return issues
    
    def _detect_outlier_issues(self, df: pd.DataFrame) -> List[Dict]:
        """Detect outlier issues"""
        issues = []
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            outliers = self._detect_outliers(df[col])
            
            if outliers["percentage"] > 5:  # More than 5% outliers
                issues.append({
                    "type": "outliers",
                    "description": f"Column '{col}' has {outliers['percentage']:.1f}% outliers",
                    "severity": "medium" if outliers["percentage"] < 10 else "high",
                    "affected_columns": [col],
                    "details": outliers
                })
        
        return issues
    
    def _synthesize_analysis(self, basic_stats: Dict, llm_analysis: Dict, 
                           rule_issues: List[Dict]) -> Dict[str, Any]:
        """Synthesize analysis results"""
        # Merge quality issues
        all_issues = rule_issues.copy()
        
        if "quality_issues" in llm_analysis:
            all_issues.extend(llm_analysis["quality_issues"])
        
        # Deduplicate and prioritize
        unique_issues = self._deduplicate_issues(all_issues)
        prioritized_issues = self._prioritize_issues(unique_issues)
        
        # Generate cleaning recommendations
        cleaning_recommendations = self._generate_cleaning_recommendations(prioritized_issues)
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(basic_stats, prioritized_issues)
        
        return {
            "basic_statistics": basic_stats,
            "quality_issues": prioritized_issues,
            "quality_score": quality_score,
            "cleaning_recommendations": cleaning_recommendations,
            "analysis_summary": {
                "total_issues": len(prioritized_issues),
                "high_severity_issues": len([i for i in prioritized_issues if i.get("severity") == "high"]),
                "estimated_cleaning_time": sum([r.get("estimated_time", 30) for r in cleaning_recommendations])
            }
        }
    
    def _deduplicate_issues(self, issues: List[Dict]) -> List[Dict]:
        """Deduplicate quality issues"""
        seen = set()
        unique_issues = []
        
        for issue in issues:
            # Create unique identifier
            identifier = (
                issue.get("type", ""),
                tuple(issue.get("affected_columns", [])),
                issue.get("description", "")
            )
            
            if identifier not in seen:
                seen.add(identifier)
                unique_issues.append(issue)
        
        return unique_issues
    
    def _prioritize_issues(self, issues: List[Dict]) -> List[Dict]:
        """Prioritize issues by severity"""
        severity_order = {"high": 3, "medium": 2, "low": 1}
        
        return sorted(issues, key=lambda x: severity_order.get(x.get("severity", "low"), 1), reverse=True)
    
    def _generate_cleaning_recommendations(self, issues: List[Dict]) -> List[Dict]:
        """Generate cleaning recommendations"""
        recommendations = []
        
        for issue in issues:
            issue_type = issue.get("type", "")
            
            if issue_type == "missing_values":
                recommendations.append({
                    "strategy": "handle_missing_values",
                    "description": "Handle missing values",
                    "methods": ["drop", "mean_fill", "median_fill", "interpolation"],
                    "estimated_time": 15,
                    "complexity": "low"
                })
            elif issue_type == "duplicate_rows":
                recommendations.append({
                    "strategy": "remove_duplicates",
                    "description": "Remove duplicate rows",
                    "methods": ["keep_first", "keep_last", "manual_selection"],
                    "estimated_time": 10,
                    "complexity": "low"
                })
            elif issue_type == "outliers":
                recommendations.append({
                    "strategy": "handle_outliers",
                    "description": "Handle outliers",
                    "methods": ["remove", "transform", "flag", "keep"],
                    "estimated_time": 30,
                    "complexity": "medium"
                })
            elif issue_type in ["type_inconsistency", "inconsistent_date_format", "case_inconsistency"]:
                recommendations.append({
                    "strategy": "standardize_format",
                    "description": "Standardize format",
                    "methods": ["type_conversion", "format_unification", "case_normalization"],
                    "estimated_time": 20,
                    "complexity": "medium"
                })
        
        return recommendations
    
    def _calculate_quality_score(self, basic_stats: Dict, issues: List[Dict]) -> float:
        """Calculate quality score"""
        if "error" in basic_stats:
            return 0.0
        
        # Base score
        base_score = 100.0
        
        # Deduct points based on issue severity
        for issue in issues:
            severity = issue.get("severity", "low")
            if severity == "high":
                base_score -= 20
            elif severity == "medium":
                base_score -= 10
            else:
                base_score -= 5
        
        # Deduct points based on missing value ratio
        missing_percentages = basic_stats.get("missing_percentage", {})
        avg_missing = sum(missing_percentages.values()) / len(missing_percentages) if missing_percentages else 0
        base_score -= avg_missing * 50  # Missing ratio * 50
        
        return max(0.0, min(100.0, base_score))
    
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
    
    def _detect_date_formats(self, series: pd.Series) -> List[str]:
        """Detect date formats"""
        formats = []
        sample_values = series.dropna().astype(str).head(100)
        
        date_patterns = [
            (r'^\d{4}-\d{2}-\d{2}$', 'YYYY-MM-DD'),
            (r'^\d{2}/\d{2}/\d{4}$', 'MM/DD/YYYY'),
            (r'^\d{2}-\d{2}-\d{4}$', 'MM-DD-YYYY'),
            (r'^\d{4}/\d{2}/\d{2}$', 'YYYY/MM/DD')
        ]
        
        for pattern, format_name in date_patterns:
            if sample_values.str.match(pattern).any():
                formats.append(format_name)
        
        return formats
    
    def _has_case_inconsistency(self, series: pd.Series) -> bool:
        """Detect case inconsistency"""
        text_values = series.dropna().astype(str)
        if len(text_values) == 0:
            return False
        
        has_upper = bool(text_values.str.contains(r'[A-Z]').any())
        has_lower = bool(text_values.str.contains(r'[a-z]').any())
        
        return has_upper and has_lower
    
    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create error response"""
        return {
            "basic_statistics": {"error": error_message},
            "quality_issues": [],
            "quality_score": 0.0,
            "cleaning_recommendations": [],
            "analysis_summary": {
                "total_issues": 0,
                "high_severity_issues": 0,
                "estimated_cleaning_time": 0
            }
        }

