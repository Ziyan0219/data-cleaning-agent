"""
Result Aggregation Agent Implementation

This module implements the result aggregation agent responsible for
collecting and synthesizing results from all other agents into a final report.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import json
from loguru import logger

from ..config.settings import get_settings


class ResultAggregationAgent:
    """Result Aggregation Agent Class"""
    
    def __init__(self, llm):
        self.llm = llm
        self.settings = get_settings()
        
        logger.info("Result Aggregation Agent initialized successfully")
    
    def aggregate_results(self, analysis_results: Dict, 
                        cleaning_results: Dict, 
                        validation_results: Dict) -> Dict[str, Any]:
        """Aggregate all results into comprehensive cattle data report"""
        logger.info("Starting cattle data results aggregation")
        
        try:
            # Generate cattle-specific report
            final_report = self._generate_cattle_report(
                analysis_results, cleaning_results, validation_results
            )
            
            # Create executive summary
            executive_summary = self._create_cattle_executive_summary(
                analysis_results, cleaning_results, validation_results
            )
            
            # Compile detailed metrics
            detailed_metrics = self._compile_cattle_metrics(
                analysis_results, cleaning_results, validation_results
            )
            
            # Prepare visualization data
            charts_data = self._prepare_cattle_visualization_data(
                analysis_results, cleaning_results, validation_results
            )
            
            result = {
                "session_id": cleaning_results.get("session_id", "unknown"),
                "final_report": final_report,
                "executive_summary": executive_summary,
                "detailed_metrics": detailed_metrics,
                "charts_data": charts_data,
                "recommendations": self._generate_cattle_recommendations(analysis_results, cleaning_results),
                "timestamp": datetime.now().isoformat(),
                "status": "completed"
            }
            
            logger.info("Cattle data results aggregation completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Results aggregation failed: {str(e)}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "status": "failed"
            }
    
    def _generate_cattle_report(self, analysis_results: Dict, cleaning_results: Dict, 
                              validation_results: Dict) -> str:
        """Generate detailed cattle data cleaning report"""
        
        # Extract cattle analysis data
        cattle_analysis = analysis_results.get("cattle_analysis", {})
        summary = cattle_analysis.get("summary", {})
        
        # Extract cleaning operations
        operation_log = cleaning_results.get("operation_log", [])
        
        report_sections = []
        
        # Header
        report_sections.append("# CATTLE DATA CLEANING REPORT")
        report_sections.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_sections.append("")
        
        # Executive Summary
        report_sections.append("## EXECUTIVE SUMMARY")
        total_lots = summary.get("total_lots", 0)
        outlier_count = summary.get("outlier_lots_count", 0)
        review_count = summary.get("review_lots_count", 0)
        label_count = summary.get("label_issues_count", 0)
        clean_count = summary.get("clean_lots_count", 0)
        
        report_sections.append(f"- **Total Lots Processed:** {total_lots}")
        report_sections.append(f"- **Clean Lots:** {clean_count}")
        report_sections.append(f"- **Issues Resolved:** {outlier_count + label_count}")
        report_sections.append(f"- **Lots Requiring Manual Review:** {review_count}")
        report_sections.append("")
        
        # Problem Classification
        report_sections.append("## PROBLEM CLASSIFICATION")
        
        if outlier_count > 0:
            report_sections.append("### 1. EXTREME OUTLIER WEIGHTS (DELETED)")
            report_sections.append(f"**Count:** {outlier_count} lots")
            report_sections.append("**Action:** Automatically deleted due to impossible weight values")
            report_sections.append("**Criteria:** Entry weight >1040 lbs OR Exit weight >1950 lbs")
            
            outlier_lots = cattle_analysis.get("outlier_lots", [])
            if outlier_lots:
                report_sections.append("**Deleted Lots:**")
                for lot in outlier_lots[:5]:  # Show first 5
                    report_sections.append(f"- {lot['lot_id']}: Entry {lot['entry_weight']} lbs, Exit {lot['exit_weight']} lbs")
                if len(outlier_lots) > 5:
                    report_sections.append(f"- ... and {len(outlier_lots) - 5} more lots")
            report_sections.append("")
        
        if review_count > 0:
            report_sections.append("### 2. QUESTIONABLE WEIGHTS (MANUAL REVIEW REQUIRED)")
            report_sections.append(f"**Count:** {review_count} lots")
            report_sections.append("**Action:** Flagged for manual review")
            report_sections.append("**Criteria:** Weights between normal range and 1.3x threshold")
            
            review_lots = cattle_analysis.get("review_lots", [])
            if review_lots:
                report_sections.append("**Lots Requiring Review:**")
                for lot in review_lots:
                    report_sections.append(f"- {lot['lot_id']}: Entry {lot['entry_weight']} lbs, Exit {lot['exit_weight']} lbs")
            report_sections.append("")
        
        if label_count > 0:
            report_sections.append("### 3. INCORRECT READY_TO_LOAD LABELS (CORRECTED)")
            report_sections.append(f"**Count:** {label_count} lots")
            report_sections.append("**Action:** Labels corrected from 'No' to 'Yes'")
            report_sections.append("**Criteria:** Normal weights but marked as not ready")
            
            label_lots = cattle_analysis.get("label_issues", [])
            if label_lots:
                report_sections.append("**Corrected Lots:**")
                for lot in label_lots:
                    report_sections.append(f"- {lot['lot_id']}: {lot['current_label']} → Yes")
            report_sections.append("")
        
        # Processing Summary
        report_sections.append("## PROCESSING SUMMARY")
        successful_ops = len([op for op in operation_log if op.get("status") == "success"])
        failed_ops = len([op for op in operation_log if op.get("status") == "failed"])
        
        report_sections.append(f"- **Successful Operations:** {successful_ops}")
        report_sections.append(f"- **Failed Operations:** {failed_ops}")
        
        if operation_log:
            report_sections.append("**Operations Performed:**")
            for op in operation_log:
                status_icon = "✅" if op.get("status") == "success" else "❌"
                report_sections.append(f"{status_icon} {op.get('description', 'Unknown operation')}")
        report_sections.append("")
        
        # Final Status
        report_sections.append("## FINAL STATUS")
        if review_count > 0:
            report_sections.append(f"⚠️  **{review_count} lots require manual review before processing**")
        else:
            report_sections.append("✅ **All lots are ready for processing**")
        
        report_sections.append(f"📊 **Data Quality Score:** {self._calculate_quality_score(summary)}%")
        report_sections.append("")
        
        # Next Steps
        if review_count > 0:
            report_sections.append("## NEXT STEPS")
            report_sections.append("1. Review flagged lots for weight accuracy")
            report_sections.append("2. Verify or correct questionable weight measurements")
            report_sections.append("3. Re-run processing after manual corrections")
        
        return "\n".join(report_sections)
    
    def _calculate_quality_score(self, summary: Dict) -> int:
        """Calculate data quality score"""
        total = summary.get("total_lots", 1)
        clean = summary.get("clean_lots_count", 0)
        corrected = summary.get("label_issues_count", 0)
        
        # Clean lots + corrected lots / total lots
        quality_score = ((clean + corrected) / total) * 100
        return int(quality_score)
    
    def _create_cattle_executive_summary(self, analysis_results: Dict, cleaning_results: Dict, 
                                       validation_results: Dict) -> str:
        """Generate executive summary for cattle data"""
        cattle_analysis = analysis_results.get("cattle_analysis", {})
        summary = cattle_analysis.get("summary", {})
        
        total_lots = summary.get("total_lots", 0)
        issues_resolved = summary.get("outlier_lots_count", 0) + summary.get("label_issues_count", 0)
        review_needed = summary.get("review_lots_count", 0)
        
        if review_needed > 0:
            status = f"Processing completed with {review_needed} lots requiring manual review"
        else:
            status = "All lots successfully processed and ready for loading"
        
        return f"Processed {total_lots} cattle lots, resolved {issues_resolved} issues automatically. {status}."
        
        try:
            # Generate final report
            final_report = self._generate_final_report(
                analysis_results, cleaning_results, validation_results
            )
            
            # Create executive summary
            executive_summary = self._create_executive_summary(
                analysis_results, cleaning_results, validation_results
            )
            
            # Compile detailed metrics
            detailed_metrics = self._compile_detailed_metrics(
                analysis_results, cleaning_results, validation_results
            )
            
            # Prepare visualization data
            charts_data = self._prepare_visualization_data(
                analysis_results, cleaning_results, validation_results
            )
            
            # Set export configuration
            export_config = self._configure_exports()
            
            result = {
                "final_report": final_report,
                "executive_summary": executive_summary,
                "detailed_metrics": detailed_metrics,
                "charts_data": charts_data,
                "visualization_config": {
                    "chart_types": ["bar", "pie", "line"],
                    "color_scheme": "professional",
                    "export_formats": ["png", "pdf", "svg"]
                },
                "export_formats": ["json", "csv", "pdf", "html"],
                "export_paths": export_config,
                "aggregation_status": "completed",
                "completion_timestamp": datetime.now()
            }
            
            logger.info("Results aggregation completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Results aggregation failed: {str(e)}")
            return {
                "final_report": {"error": str(e)},
                "executive_summary": f"Aggregation failed: {str(e)}",
                "detailed_metrics": {},
                "charts_data": [],
                "aggregation_status": "failed",
                "completion_timestamp": datetime.now(),
                "error": str(e)
            }
    
    def generate_report(self, aggregated_results: Dict) -> str:
        """Generate cleaning report - implement based on your specific needs"""
        try:
            report_sections = []
            
            # Title and metadata
            report_sections.append("# Data Cleaning Report")
            report_sections.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_sections.append("")
            
            # Executive summary
            if "executive_summary" in aggregated_results:
                report_sections.append("## Executive Summary")
                report_sections.append(aggregated_results["executive_summary"])
                report_sections.append("")
            
            # Detailed metrics
            if "detailed_metrics" in aggregated_results:
                report_sections.append("## Quality Metrics")
                metrics = aggregated_results["detailed_metrics"]
                for category, values in metrics.items():
                    report_sections.append(f"### {category.replace('_', ' ').title()}")
                    if isinstance(values, dict):
                        for key, value in values.items():
                            report_sections.append(f"- {key}: {value}")
                    report_sections.append("")
            
            # Final report details
            if "final_report" in aggregated_results:
                report_sections.append("## Detailed Analysis")
                final_report = aggregated_results["final_report"]
                for section, content in final_report.items():
                    if section != "error":
                        report_sections.append(f"### {section.replace('_', ' ').title()}")
                        if isinstance(content, (list, dict)):
                            report_sections.append(f"```json\n{json.dumps(content, indent=2)}\n```")
                        else:
                            report_sections.append(str(content))
                        report_sections.append("")
            
            return "\n".join(report_sections)
            
        except Exception as e:
            logger.error(f"Report generation failed: {str(e)}")
            return f"Report generation failed: {str(e)}"
    
    def _generate_final_report(self, analysis_results: Dict, 
                             cleaning_results: Dict, 
                             validation_results: Dict) -> Dict[str, Any]:
        """Generate final comprehensive report"""
        report = {
            "data_analysis_summary": self._summarize_analysis(analysis_results),
            "cleaning_operations_summary": self._summarize_cleaning(cleaning_results),
            "validation_summary": self._summarize_validation(validation_results),
            "overall_assessment": self._create_overall_assessment(
                analysis_results, cleaning_results, validation_results
            ),
            "recommendations": self._compile_recommendations(
                analysis_results, cleaning_results, validation_results
            )
        }
        
        return report
    
    def _create_executive_summary(self, analysis_results: Dict, 
                                cleaning_results: Dict, 
                                validation_results: Dict) -> str:
        """Create executive summary"""
        try:
            # Extract key metrics
            total_issues = analysis_results.get("analysis_summary", {}).get("total_issues", 0)
            quality_score = analysis_results.get("quality_score", 0)
            overall_validation_score = validation_results.get("overall_score", 0)
            
            # Extract cleaning summary
            cleaning_summary = cleaning_results.get("cleaning_summary", {})
            successful_operations = cleaning_summary.get("successful_operations", 0)
            total_operations = cleaning_summary.get("total_operations", 0)
            
            summary = f"""
Data Cleaning Process Summary:

• Initial Quality Assessment: {total_issues} issues identified with quality score of {quality_score:.1f}/100
• Cleaning Operations: {successful_operations}/{total_operations} operations completed successfully
• Final Quality Score: {overall_validation_score:.1f}/100
• Overall Status: {'Success' if overall_validation_score >= 0.7 else 'Needs Improvement'}

The data cleaning process has {'significantly improved' if overall_validation_score > quality_score/100 else 'maintained'} the overall data quality.
Key improvements include enhanced completeness, reduced duplicates, and standardized formats.
            """.strip()
            
            return summary
            
        except Exception as e:
            logger.error(f"Error creating executive summary: {str(e)}")
            return f"Executive summary generation failed: {str(e)}"
    
    def _compile_detailed_metrics(self, analysis_results: Dict, 
                                cleaning_results: Dict, 
                                validation_results: Dict) -> Dict[str, Any]:
        """Compile detailed metrics"""
        metrics = {
            "quality_metrics": validation_results.get("quality_scores", {}),
            "improvement_metrics": cleaning_results.get("improvement_metrics", {}),
            "operation_metrics": {
                "total_operations": cleaning_results.get("cleaning_summary", {}).get("total_operations", 0),
                "successful_operations": cleaning_results.get("cleaning_summary", {}).get("successful_operations", 0),
                "failed_operations": cleaning_results.get("cleaning_summary", {}).get("failed_operations", 0)
            },
            "data_statistics": analysis_results.get("basic_statistics", {}),
            "validation_tests": validation_results.get("validation_tests", [])
        }
        
        return metrics
    
    def _prepare_visualization_data(self, analysis_results: Dict, 
                                  cleaning_results: Dict, 
                                  validation_results: Dict) -> List[Dict]:
        """Prepare data for visualization"""
        charts = []
        
        try:
            # Quality metrics chart
            quality_scores = validation_results.get("quality_scores", {})
            if quality_scores:
                charts.append({
                    "chart_type": "bar",
                    "title": "Quality Metrics",
                    "data": {
                        "labels": list(quality_scores.keys()),
                        "values": list(quality_scores.values())
                    },
                    "config": {
                        "y_axis_label": "Score",
                        "x_axis_label": "Metrics"
                    }
                })
            
            # Before/after comparison
            comparison = validation_results.get("before_after_comparison", {})
            if comparison:
                charts.append({
                    "chart_type": "line",
                    "title": "Before vs After Comparison",
                    "data": {
                        "categories": list(comparison.keys()),
                        "before": [comp.get("before", 0) for comp in comparison.values() if isinstance(comp, dict)],
                        "after": [comp.get("after", 0) for comp in comparison.values() if isinstance(comp, dict)]
                    }
                })
            
            # Issue distribution
            quality_issues = analysis_results.get("quality_issues", [])
            if quality_issues:
                issue_types = {}
                for issue in quality_issues:
                    issue_type = issue.get("type", "unknown")
                    issue_types[issue_type] = issue_types.get(issue_type, 0) + 1
                
                charts.append({
                    "chart_type": "pie",
                    "title": "Issue Distribution",
                    "data": {
                        "labels": list(issue_types.keys()),
                        "values": list(issue_types.values())
                    }
                })
            
        except Exception as e:
            logger.error(f"Error preparing visualization data: {str(e)}")
        
        return charts
    
    def _configure_exports(self) -> Dict[str, str]:
        """Configure export paths"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        return {
            "json": f"data/output/cleaning_report_{timestamp}.json",
            "csv": f"data/output/cleaning_metrics_{timestamp}.csv",
            "pdf": f"data/output/cleaning_report_{timestamp}.pdf",
            "html": f"data/output/cleaning_report_{timestamp}.html"
        }
    
    def _summarize_analysis(self, analysis_results: Dict) -> Dict[str, Any]:
        """Summarize analysis results"""
        return {
            "total_issues_found": len(analysis_results.get("quality_issues", [])),
            "quality_score": analysis_results.get("quality_score", 0),
            "high_priority_issues": len([
                issue for issue in analysis_results.get("quality_issues", [])
                if issue.get("severity") == "high"
            ]),
            "data_statistics": analysis_results.get("basic_statistics", {})
        }
    
    def _summarize_cleaning(self, cleaning_results: Dict) -> Dict[str, Any]:
        """Summarize cleaning results"""
        return {
            "operations_performed": len(cleaning_results.get("operation_log", [])),
            "successful_operations": len(cleaning_results.get("completed_operations", [])),
            "failed_operations": len(cleaning_results.get("failed_operations", [])),
            "improvement_metrics": cleaning_results.get("improvement_metrics", {})
        }
    
    def _summarize_validation(self, validation_results: Dict) -> Dict[str, Any]:
        """Summarize validation results"""
        return {
            "overall_score": validation_results.get("overall_score", 0),
            "pass_fail_status": validation_results.get("pass_fail_status", "unknown"),
            "quality_scores": validation_results.get("quality_scores", {}),
            "validation_tests_passed": len([
                test for test in validation_results.get("validation_tests", [])
                if test.get("status") == "pass"
            ])
        }
    
    def _create_overall_assessment(self, analysis_results: Dict, 
                                 cleaning_results: Dict, 
                                 validation_results: Dict) -> Dict[str, Any]:
        """Create overall assessment"""
        overall_score = validation_results.get("overall_score", 0)
        
        if overall_score >= 0.9:
            assessment = "Excellent"
            description = "Data quality is excellent with minimal issues remaining."
        elif overall_score >= 0.7:
            assessment = "Good"
            description = "Data quality is good with some minor issues that may need attention."
        elif overall_score >= 0.5:
            assessment = "Fair"
            description = "Data quality is fair but requires additional cleaning efforts."
        else:
            assessment = "Poor"
            description = "Data quality is poor and requires significant improvement."
        
        return {
            "assessment": assessment,
            "description": description,
            "overall_score": overall_score,
            "confidence_level": "high" if overall_score >= 0.7 else "medium"
        }
    
    def _compile_recommendations(self, analysis_results: Dict, 
                               cleaning_results: Dict, 
                               validation_results: Dict) -> List[str]:
        """Compile recommendations from all agents"""
        recommendations = []
        
        # From analysis
        analysis_recommendations = analysis_results.get("cleaning_recommendations", [])
        for rec in analysis_recommendations:
            if isinstance(rec, dict):
                recommendations.append(rec.get("description", str(rec)))
            else:
                recommendations.append(str(rec))
        
        # From validation
        validation_recommendations = validation_results.get("recommendations", [])
        recommendations.extend(validation_recommendations)
        
        # Add general recommendations based on overall assessment
        overall_score = validation_results.get("overall_score", 0)
        if overall_score < 0.7:
            recommendations.append("Consider additional cleaning iterations to improve data quality")
        
        if not recommendations:
            recommendations.append("Data quality is satisfactory. No additional actions required.")
        
        return list(set(recommendations))  # Remove duplicates

