"""
Main Controller Agent Implementation

This module implements the main controller agent that orchestrates the entire
data cleaning workflow using LangGraph's state graph architecture.
"""

import asyncio
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
from loguru import logger

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

from ..schemas.state import (
    DataCleaningState, 
    create_initial_state, 
    update_state_phase
)
from ..config.settings import get_settings
from .data_analysis_agent import DataAnalysisAgent
from .data_cleaning_agent import DataCleaningAgent
from .quality_validation_agent import QualityValidationAgent
from .result_aggregation_agent import ResultAggregationAgent


class MainControllerAgent:
    """Main Controller Agent Class"""
    
    def __init__(self):
        self.settings = get_settings()
        self.session_id = str(uuid.uuid4())
        self.workflow = None
        self.llm = self._initialize_llm()
        
        # Initialize specialized agents
        self.analysis_agent = DataAnalysisAgent(self.llm)
        self.cleaning_agent = DataCleaningAgent(self.llm)
        self.validation_agent = QualityValidationAgent(self.llm)
        self.aggregation_agent = ResultAggregationAgent(self.llm)
        
        # Build workflow
        self._build_workflow()
        
        logger.info("Main Controller Agent initialized successfully")
    
    def _initialize_llm(self):
        """Initialize LLM"""
        llm_config = self.settings.get_llm_config()
        
        return ChatOpenAI(
            model=llm_config.model,
            api_key=llm_config.api_key,
            temperature=llm_config.temperature,
            max_tokens=llm_config.max_tokens,
            timeout=llm_config.timeout,
            max_retries=llm_config.max_retries
        )
    
    def _build_workflow(self):
        """Build LangGraph workflow"""
        # Create state graph
        workflow = StateGraph(DataCleaningState)
        
        # Add nodes
        workflow.add_node("load_data", self._load_data)
        workflow.add_node("analyze_data", self._analyze_data)
        workflow.add_node("plan_cleaning", self._plan_cleaning)
        workflow.add_node("execute_cleaning", self._execute_cleaning)
        workflow.add_node("validate_results", self._validate_results)
        workflow.add_node("aggregate_results", self._aggregate_results)
        workflow.add_node("handle_error", self._handle_error)
        
        # Set entry point
        workflow.set_entry_point("load_data")
        
        # Add edges
        workflow.add_edge("load_data", "analyze_data")
        workflow.add_edge("analyze_data", "plan_cleaning")
        workflow.add_edge("plan_cleaning", "execute_cleaning")
        workflow.add_edge("execute_cleaning", "validate_results")
        workflow.add_edge("validate_results", "aggregate_results")
        workflow.add_edge("aggregate_results", END)
        
        # Add conditional edges for error handling
        workflow.add_conditional_edges(
            "load_data",
            self._should_continue_after_load,
            {
                "continue": "analyze_data",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "analyze_data",
            self._should_continue_after_analysis,
            {
                "continue": "plan_cleaning",
                "retry": "analyze_data",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "execute_cleaning",
            self._should_continue_after_cleaning,
            {
                "continue": "validate_results",
                "retry": "execute_cleaning",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "validate_results",
            self._should_continue_after_validation,
            {
                "continue": "aggregate_results",
                "retry_cleaning": "execute_cleaning",
                "error": "handle_error"
            }
        )
        
        workflow.add_edge("handle_error", END)
        
        # Compile workflow
        self.workflow = workflow.compile()
        
        logger.info("Workflow built successfully")
    
    async def process_data(self, user_requirements: str, 
                          data_source: str) -> Dict[str, Any]:
        """Process data cleaning request"""
        try:
            logger.info(f"Starting data cleaning process for session: {self.session_id}")
            
            # Create initial state
            initial_state = create_initial_state(
                session_id=self.session_id,
                user_requirements=user_requirements,
                data_source=data_source
            )
            
            # Execute workflow
            start_time = datetime.now()
            final_state = await self.workflow.ainvoke(initial_state)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Prepare result
            result = {
                "session_id": self.session_id,
                "status": "completed" if final_state["current_phase"] != "error" else "failed",
                "execution_time": execution_time,
                "results": final_state.get("agent_results", {}),
                "quality_metrics": final_state.get("quality_metrics", {}),
                "final_data": final_state.get("processed_data"),
                "error": final_state.get("error_log", [])[-1] if final_state.get("error_log") else None
            }
            
            logger.info(f"Data cleaning process completed: {result['status']}")
            return result
            
        except Exception as e:
            logger.error(f"Error in data cleaning process: {str(e)}")
            return {
                "session_id": self.session_id,
                "status": "failed",
                "error": str(e),
                "execution_time": 0,
                "results": {},
                "quality_metrics": {},
                "final_data": None
            }
    
    def _load_data(self, state: DataCleaningState) -> Dict[str, Any]:
        """Load data from source"""
        logger.info("Loading data from source")
        
        try:
            data_source = state["data_source"]
            
            # Determine data format
            if data_source.endswith('.csv'):
                data_format = "csv"
            elif data_source.endswith(('.xlsx', '.xls')):
                data_format = "excel"
            elif data_source.endswith('.json'):
                data_format = "json"
            else:
                # Assume it's raw data content
                data_format = "raw"
            
            # Load data based on format
            if data_format in ["csv", "excel", "json"]:
                # Load from file
                with open(data_source, 'r', encoding='utf-8') as f:
                    raw_data = f.read()
            else:
                # Use as raw data
                raw_data = data_source
            
            logger.info(f"Data loaded successfully: {len(raw_data)} characters")
            
            # Return only the fields we want to update
            return {
                "data_format": data_format,
                "raw_data": raw_data,
                "current_phase": "data_loaded",
                "progress_percentage": 10.0
            }
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            error_entry = {
                "phase": "load_data",
                "error": str(e),
                "timestamp": datetime.now()
            }
            
            return {
                "current_phase": "error",
                "progress_percentage": 0.0,
                "error_log": state.get("error_log", []) + [error_entry]
            }
    
    def _analyze_data(self, state: DataCleaningState) -> Dict[str, Any]:
        """Analyze data quality"""
        logger.info("Analyzing data quality")
        
        try:
            # Call data analysis agent
            analysis_result = self.analysis_agent.analyze_data_quality(
                data=state["raw_data"],
                user_requirements=state["user_requirements"]
            )
            
            logger.info(f"Data analysis completed: {len(analysis_result.get('quality_issues', []))} issues found")
            
            # Return only the fields we want to update
            agent_results = state.get("agent_results", {})
            agent_results["analysis"] = analysis_result
            
            return {
                "agent_results": agent_results,
                "quality_issues": analysis_result.get("quality_issues", []),
                "quality_metrics": analysis_result.get("quality_metrics", {}),
                "data_statistics": analysis_result.get("basic_statistics", {}),
                "current_phase": "analysis_completed",
                "progress_percentage": 30.0
            }
            
        except Exception as e:
            logger.error(f"Error in data analysis: {str(e)}")
            error_entry = {
                "phase": "analyze_data",
                "error": str(e),
                "timestamp": datetime.now()
            }
            
            return {
                "current_phase": "error",
                "progress_percentage": 0.0,
                "error_log": state.get("error_log", []) + [error_entry]
            }
    
    def _plan_cleaning(self, state: DataCleaningState) -> Dict[str, Any]:
        """Plan cleaning operations"""
        logger.info("Planning cleaning operations")
        
        try:
            # Generate cleaning plan based on analysis
            analysis_result = state["agent_results"].get("analysis", {})
            
            # Simple cleaning plan for cattle data
            cleaning_plan = {
                "operations": [
                    {
                        "type": "outlier_removal",
                        "description": "Remove extreme weight outliers",
                        "strategy": "remove_outliers"
                    },
                    {
                        "type": "manual_review",
                        "description": "Flag questionable weights for review",
                        "strategy": "manual_review"
                    },
                    {
                        "type": "label_correction",
                        "description": "Correct ready_to_load labels",
                        "strategy": "correct_labels"
                    }
                ]
            }
            
            logger.info(f"Cleaning plan created with {len(cleaning_plan['operations'])} operations")
            
            return {
                "execution_plan": cleaning_plan["operations"],
                "current_phase": "planning_completed",
                "progress_percentage": 50.0
            }
            
        except Exception as e:
            logger.error(f"Error in planning: {str(e)}")
            error_entry = {
                "phase": "plan_cleaning",
                "error": str(e),
                "timestamp": datetime.now()
            }
            
            return {
                "current_phase": "error",
                "progress_percentage": 0.0,
                "error_log": state.get("error_log", []) + [error_entry]
            }
    
    def _plan_cleaning(self, state: DataCleaningState) -> DataCleaningState:
        """Plan cleaning operations"""
        logger.info("Planning cleaning operations")
        
        try:
            # Generate cleaning plan based on analysis results
            quality_issues = state["quality_issues"]
            user_requirements = state["user_requirements"]
            
            # Create execution plan
            execution_plan = []
            
            for issue in quality_issues:
                operation = {
                    "operation_id": str(uuid.uuid4()),
                    "issue_type": issue.get("type"),
                    "description": issue.get("description"),
                    "affected_columns": issue.get("affected_columns", []),
                    "severity": issue.get("severity", "medium"),
                    "strategy": self._determine_cleaning_strategy(issue),
                    "priority": self._calculate_priority(issue)
                }
                execution_plan.append(operation)
            
            # Sort by priority
            execution_plan.sort(key=lambda x: x["priority"], reverse=True)
            
            state["execution_plan"] = execution_plan
            state = update_state_phase(state, "planning_completed", 40.0)
            
            logger.info(f"Cleaning plan created: {len(execution_plan)} operations")
            
        except Exception as e:
            logger.error(f"Error in planning: {str(e)}")
            state["error_log"].append({
                "phase": "plan_cleaning",
                "error": str(e),
                "timestamp": datetime.now()
            })
        
        return state
    
    def _execute_cleaning(self, state: DataCleaningState) -> DataCleaningState:
        """Execute cleaning operations"""
        logger.info("Executing cleaning operations")
        
        try:
            # Call data cleaning agent
            cleaning_result = self.cleaning_agent.execute_cleaning_plan(
                data=state["raw_data"],
                execution_plan=state["execution_plan"],
                config=state.get("cleaning_config", {})
            )
            
            # Update state with cleaning results
            state["agent_results"]["cleaning"] = cleaning_result
            state["processed_data"] = cleaning_result.get("cleaned_data")
            state["completed_tasks"] = cleaning_result.get("completed_operations", [])
            
            state = update_state_phase(state, "cleaning_completed", 70.0)
            
            logger.info("Data cleaning completed successfully")
            
        except Exception as e:
            logger.error(f"Error in cleaning execution: {str(e)}")
            state["error_log"].append({
                "phase": "execute_cleaning",
                "error": str(e),
                "timestamp": datetime.now()
            })
            state["retry_count"] += 1
        
        return state
    
    def _validate_results(self, state: DataCleaningState) -> DataCleaningState:
        """Validate cleaning results"""
        logger.info("Validating cleaning results")
        
        try:
            # Call quality validation agent
            validation_result = self.validation_agent.validate_cleaning_results(
                original_data=state["raw_data"],
                cleaned_data=state["processed_data"],
                cleaning_log=state["agent_results"].get("cleaning", {})
            )
            
            # Update state with validation results
            state["agent_results"]["validation"] = validation_result
            state["validation_results"] = validation_result
            
            # Update quality metrics
            if "quality_scores" in validation_result:
                state["quality_metrics"].update(validation_result["quality_scores"])
            
            state = update_state_phase(state, "validation_completed", 85.0)
            
            logger.info("Results validation completed")
            
        except Exception as e:
            logger.error(f"Error in validation: {str(e)}")
            state["error_log"].append({
                "phase": "validate_results",
                "error": str(e),
                "timestamp": datetime.now()
            })
        
        return state
    
    def _aggregate_results(self, state: DataCleaningState) -> DataCleaningState:
        """Aggregate final results"""
        logger.info("Aggregating final results")
        
        try:
            # Call result aggregation agent
            aggregation_result = self.aggregation_agent.aggregate_results(
                analysis_results=state["agent_results"].get("analysis", {}),
                cleaning_results=state["agent_results"].get("cleaning", {}),
                validation_results=state["agent_results"].get("validation", {})
            )
            
            # Update state with aggregated results
            state["agent_results"]["aggregation"] = aggregation_result
            
            state = update_state_phase(state, "completed", 100.0)
            
            logger.info("Results aggregation completed")
            
        except Exception as e:
            logger.error(f"Error in aggregation: {str(e)}")
            state["error_log"].append({
                "phase": "aggregate_results",
                "error": str(e),
                "timestamp": datetime.now()
            })
        
        return state
    
    def _handle_error(self, state: DataCleaningState) -> DataCleaningState:
        """Handle errors"""
        logger.error("Handling error in workflow")
        
        state = update_state_phase(state, "error", state["progress_percentage"])
        
        # Add error summary
        if state["error_log"]:
            latest_error = state["error_log"][-1]
            logger.error(f"Latest error: {latest_error}")
        
        return state
    
    # Conditional edge functions
    def _should_continue_after_load(self, state: DataCleaningState) -> str:
        """Check if should continue after data loading"""
        if state["current_phase"] == "error":
            return "error"
        return "continue"
    
    def _should_continue_after_analysis(self, state: DataCleaningState) -> str:
        """Check if should continue after analysis"""
        if state["current_phase"] == "error":
            return "error"
        
        # Check if retry is needed
        if state["retry_count"] > 0 and state["retry_count"] < self.settings.agent.retry_attempts:
            return "retry"
        
        return "continue"
    
    def _should_continue_after_cleaning(self, state: DataCleaningState) -> str:
        """Check if should continue after cleaning"""
        if state["current_phase"] == "error":
            return "error"
        
        # Check if retry is needed
        if state["retry_count"] > 0 and state["retry_count"] < self.settings.agent.retry_attempts:
            return "retry"
        
        return "continue"
    
    def _should_continue_after_validation(self, state: DataCleaningState) -> str:
        """Check if should continue after validation"""
        if state["current_phase"] == "error":
            return "error"
        
        # Check if cleaning needs to be retried based on validation results
        validation_results = state.get("validation_results", {})
        overall_score = validation_results.get("overall_score", 0)
        
        if overall_score < 0.7 and state["retry_count"] < self.settings.agent.retry_attempts:
            return "retry_cleaning"
        
        return "continue"
    
    # Helper functions
    def _determine_cleaning_strategy(self, issue: Dict) -> str:
        """Determine cleaning strategy for an issue"""
        issue_type = issue.get("type", "")
        
        strategy_mapping = {
            "missing_values": "fill_mean",
            "duplicates": "remove_duplicates",
            "outliers": "flag_outliers",
            "format_inconsistency": "standardize_format",
            "type_inconsistency": "convert_type"
        }
        
        return strategy_mapping.get(issue_type, "manual_review")
    
    def _calculate_priority(self, issue: Dict) -> int:
        """Calculate priority for an issue"""
        severity = issue.get("severity", "medium")
        
        priority_mapping = {
            "high": 3,
            "medium": 2,
            "low": 1
        }
        
        return priority_mapping.get(severity, 2)


# Convenience function for direct usage
async def process_cleaning_request(user_requirements: str, 
                                 data_source: str) -> Dict[str, Any]:
    """Process a data cleaning request"""
    controller = MainControllerAgent()
    return await controller.process_data(user_requirements, data_source)

