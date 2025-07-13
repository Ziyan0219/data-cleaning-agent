"""
Data Cleaning Agent State Definitions

This module defines various state structures used in the data cleaning workflow,
based on TypedDict to ensure type safety and data consistency.
"""

from typing import TypedDict, List, Dict, Optional, Any, Annotated
from datetime import datetime
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


class DataCleaningState(TypedDict):
    """Main data cleaning workflow state"""
    # Session information - only set once by main controller
    session_id: str
    user_id: Optional[str]
    created_at: datetime
    
    # User requirements
    user_requirements: str
    cleaning_objectives: List[str]
    quality_thresholds: Dict[str, float]
    
    # Data information
    data_source: str
    data_format: str
    raw_data: Optional[str]
    processed_data: Optional[str]
    data_schema: Optional[Dict]
    data_statistics: Optional[Dict]
    
    # Execution status
    current_phase: str
    execution_plan: List[Dict]
    completed_tasks: List[str]
    pending_tasks: List[str]
    progress_percentage: float
    
    # Agent communication - use add_messages for concurrent updates
    messages: Annotated[List[BaseMessage], add_messages]
    agent_results: Dict[str, Dict]
    communication_log: List[Dict]
    
    # Quality control
    quality_issues: List[Dict]
    quality_metrics: Dict[str, float]
    validation_results: Dict[str, Any]
    improvement_metrics: Dict[str, float]
    
    # Error handling
    error_log: List[Dict]
    warnings: List[str]
    retry_count: int
    
    # Configuration parameters
    cleaning_config: Dict[str, Any]
    llm_config: Dict[str, Any]
    agent_config: Dict[str, Any]


class AnalysisState(TypedDict):
    """Data Analysis Agent state"""
    analysis_id: str
    data_sample: str
    schema_info: Dict
    
    # Analysis results
    basic_statistics: Dict[str, Any]
    quality_assessment: Dict[str, Any]
    issue_detection: List[Dict]
    data_profiling: Dict[str, Any]
    
    # Recommendations
    cleaning_recommendations: List[Dict]
    priority_ranking: List[str]
    estimated_effort: Dict[str, int]
    
    # Status
    analysis_status: str
    analysis_timestamp: datetime
    confidence_scores: Dict[str, float]


class CleaningState(TypedDict):
    """Data Cleaning Agent state"""
    cleaning_id: str
    input_data: str
    cleaning_plan: Dict[str, Any]
    
    # Cleaning process
    current_step: str
    applied_operations: List[Dict]
    intermediate_results: List[str]
    
    # Cleaning results
    cleaned_data: Optional[str]
    cleaning_summary: Dict[str, Any]
    quality_improvement: Dict[str, float]
    
    # Metadata
    processing_time: float
    memory_usage: float
    operation_count: int
    
    # Status
    cleaning_status: str
    error_details: Optional[Dict]


class ValidationState(TypedDict):
    """Quality Validation Agent state"""
    validation_id: str
    original_data: str
    cleaned_data: str
    cleaning_log: Dict[str, Any]
    
    # Validation results
    quality_scores: Dict[str, float]
    validation_tests: List[Dict]
    anomaly_detection: Dict[str, Any]
    
    # Assessment
    overall_score: float
    pass_fail_status: str
    recommendations: List[str]
    
    # Comparison analysis
    before_after_comparison: Dict[str, Any]
    improvement_analysis: Dict[str, Any]
    
    # Status
    validation_status: str
    validation_timestamp: datetime


class AggregationState(TypedDict):
    """Result Aggregation Agent state"""
    aggregation_id: str
    
    # Input data
    analysis_results: Dict[str, Any]
    cleaning_results: Dict[str, Any]
    validation_results: Dict[str, Any]
    
    # Aggregated results
    final_report: Dict[str, Any]
    executive_summary: str
    detailed_metrics: Dict[str, Any]
    
    # Visualization
    charts_data: List[Dict]
    visualization_config: Dict[str, Any]
    
    # Export
    export_formats: List[str]
    export_paths: Dict[str, str]
    
    # Status
    aggregation_status: str
    completion_timestamp: datetime


class AgentMessage(TypedDict):
    """Inter-agent communication message"""
    message_id: str
    sender: str
    receiver: str
    message_type: str
    content: Any
    timestamp: datetime
    correlation_id: str
    priority: int
    metadata: Optional[Dict]


class ErrorInfo(TypedDict):
    """Error information structure"""
    error_id: str
    error_type: str
    error_message: str
    error_details: Dict[str, Any]
    timestamp: datetime
    context: Dict[str, Any]
    stack_trace: Optional[str]
    recovery_suggestions: List[str]


class QualityMetrics(TypedDict):
    """Data quality metrics"""
    completeness: float      # Completeness (0-1)
    accuracy: float         # Accuracy (0-1)
    consistency: float      # Consistency (0-1)
    validity: float         # Validity (0-1)
    uniqueness: float       # Uniqueness (0-1)
    timeliness: float       # Timeliness (0-1)
    overall_score: float    # Overall score (0-1)


class CleaningOperation(TypedDict):
    """Cleaning operation definition"""
    operation_id: str
    operation_type: str
    operation_name: str
    description: str
    parameters: Dict[str, Any]
    target_columns: List[str]
    conditions: Optional[Dict[str, Any]]
    expected_outcome: str
    priority: int
    estimated_time: float


# State transition functions
def create_initial_state(session_id: str, user_requirements: str, 
                        data_source: str) -> DataCleaningState:
    """Create initial state"""
    return DataCleaningState(
        session_id=session_id,
        user_id=None,
        created_at=datetime.now(),
        user_requirements=user_requirements,
        cleaning_objectives=[],
        quality_thresholds={},
        data_source=data_source,
        data_format="",
        raw_data=None,
        processed_data=None,
        data_schema=None,
        data_statistics=None,
        current_phase="initialization",
        execution_plan=[],
        completed_tasks=[],
        pending_tasks=[],
        progress_percentage=0.0,
        messages=[],
        agent_results={},
        communication_log=[],
        quality_issues=[],
        quality_metrics={},
        validation_results={},
        improvement_metrics={},
        error_log=[],
        warnings=[],
        retry_count=0,
        cleaning_config={},
        llm_config={},
        agent_config={}
    )


def update_state_phase(state: DataCleaningState, new_phase: str, 
                      progress: float) -> DataCleaningState:
    """Update state phase"""
    state["current_phase"] = new_phase
    state["progress_percentage"] = progress
    return state


def add_quality_issue(state: DataCleaningState, issue: Dict) -> DataCleaningState:
    """Add quality issue"""
    state["quality_issues"].append(issue)
    return state


def update_quality_metrics(state: DataCleaningState, 
                          metrics: QualityMetrics) -> DataCleaningState:
    """Update quality metrics"""
    state["quality_metrics"].update(metrics)
    return state

