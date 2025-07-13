"""
Main Controller Agent Implementation

This module implements the central orchestrator for the data cleaning system.
It coordinates all other agents and manages the overall workflow.
"""

import uuid
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from loguru import logger

from ..schemas.state import DataCleaningState, create_initial_state
from ..config.settings import get_settings
from .data_analysis_agent import DataAnalysisAgent
from .data_cleaning_agent import DataCleaningAgent
from .quality_validation_agent import QualityValidationAgent
from .result_aggregation_agent import ResultAggregationAgent


class MainControllerAgent:
    """Main controller for data cleaning workflow"""
    
    def __init__(self):
        """Initialize the main controller"""
        self.settings = get_settings()
        self.session_id = str(uuid.uuid4())
        
        # Initialize sub-agents
        self.analysis_agent = DataAnalysisAgent()
        self.cleaning_agent = DataCleaningAgent()
        self.validation_agent = QualityValidationAgent()
        self.aggregation_agent = ResultAggregationAgent()
        
        # Build workflow
        self.workflow = self._build_workflow()
        
        logger.info("Main Controller Agent initialized successfully")
    
    def _get_llm(self):
        """Get configured LLM instance"""
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
        
        # Set entry point
        workflow.set_entry_point("load_data")
        
        # Add edges
        workflow.add_edge("load_data", "analyze_data")
        workflow.add_edge("analyze_data", "plan_cleaning")
        workflow.add_edge("plan_cleaning", "execute_cleaning")
        workflow.add_edge("execute_cleaning", "validate_results")
        workflow.add_edge("validate_results", "aggregate_results")
        workflow.add_edge("aggregate_results", END)
        
        return workflow.compile()
    
    async def process_cleaning_request(self, user_requirements: str, 
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
            
            start_time = datetime.now()
            
            # Execute workflow
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
                "final_report": final_state.get("agent_results", {}).get("aggregation", {}).get("final_report", ""),
                "executive_summary": final_state.get("agent_results", {}).get("aggregation", {}).get("executive_summary", ""),
                "detailed_metrics": final_state.get("agent_results", {}).get("aggregation", {}).get("detailed_metrics", {}),
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
                "final_data": None,
                "final_report": "",
                "executive_summary": "",
                "detailed_metrics": {}
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
                data_format = "raw"
            
            # Load data based on format
            if data_format in ["csv", "excel", "json"]:
                with open(data_source, 'r', encoding='utf-8') as f:
                    raw_data = f.read()
            else:
                raw_data = data_source
            
            logger.info(f"Data loaded successfully: {len(raw_data)} characters")
            
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
            analysis_result = self.analysis_agent.analyze_data_quality(
                data=state["raw_data"],
                user_requirements=state["user_requirements"]
            )
            
            logger.info(f"Data analysis completed: {len(analysis_result.get('quality_issues', []))} issues found")
            
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
    
    def _execute_cleaning(self, state: DataCleaningState) -> Dict[str, Any]:
        """Execute cleaning operations"""
        logger.info("Executing cleaning operations")
        
        try:
            cleaning_result = self.cleaning_agent.clean_data(
                data=state["raw_data"],
                cleaning_plan={"operations": state["execution_plan"]},
                user_requirements=state["user_requirements"]
            )
            
            logger.info("Data cleaning completed successfully")
            
            agent_results = state.get("agent_results", {})
            agent_results["cleaning"] = cleaning_result
            
            return {
                "agent_results": agent_results,
                "processed_data": cleaning_result.get("cleaned_data"),
                "current_phase": "cleaning_completed",
                "progress_percentage": 70.0
            }
            
        except Exception as e:
            logger.error(f"Error in cleaning: {str(e)}")
            error_entry = {
                "phase": "execute_cleaning",
                "error": str(e),
                "timestamp": datetime.now()
            }
            
            return {
                "current_phase": "error",
                "progress_percentage": 0.0,
                "error_log": state.get("error_log", []) + [error_entry]
            }
    
    def _validate_results(self, state: DataCleaningState) -> Dict[str, Any]:
        """Validate cleaning results"""
        logger.info("Validating cleaning results")
        
        try:
            validation_result = self.validation_agent.validate_quality(
                original_data=state["raw_data"],
                cleaned_data=state["processed_data"],
                cleaning_log=state["agent_results"].get("cleaning", {}),
                user_requirements=state["user_requirements"]
            )
            
            logger.info("Data validation completed successfully")
            
            agent_results = state.get("agent_results", {})
            agent_results["validation"] = validation_result
            
            return {
                "agent_results": agent_results,
                "validation_results": validation_result,
                "current_phase": "validation_completed",
                "progress_percentage": 90.0
            }
            
        except Exception as e:
            logger.error(f"Error in validation: {str(e)}")
            error_entry = {
                "phase": "validate_results",
                "error": str(e),
                "timestamp": datetime.now()
            }
            
            return {
                "current_phase": "error",
                "progress_percentage": 0.0,
                "error_log": state.get("error_log", []) + [error_entry]
            }
    
    def _aggregate_results(self, state: DataCleaningState) -> Dict[str, Any]:
        """Aggregate all results"""
        logger.info("Aggregating results")
        
        try:
            aggregation_result = self.aggregation_agent.aggregate_results(
                analysis_results=state["agent_results"].get("analysis", {}),
                cleaning_results=state["agent_results"].get("cleaning", {}),
                validation_results=state["agent_results"].get("validation", {})
            )
            
            logger.info("Results aggregation completed successfully")
            
            agent_results = state.get("agent_results", {})
            agent_results["aggregation"] = aggregation_result
            
            return {
                "agent_results": agent_results,
                "current_phase": "completed",
                "progress_percentage": 100.0
            }
            
        except Exception as e:
            logger.error(f"Error in aggregation: {str(e)}")
            error_entry = {
                "phase": "aggregate_results",
                "error": str(e),
                "timestamp": datetime.now()
            }
            
            return {
                "current_phase": "error",
                "progress_percentage": 0.0,
                "error_log": state.get("error_log", []) + [error_entry]
            }


# Global function for external use
async def process_cleaning_request(user_requirements: str, data_source: str) -> Dict[str, Any]:
    """Process cleaning request using main controller"""
    controller = MainControllerAgent()
    return await controller.process_cleaning_request(user_requirements, data_source)

