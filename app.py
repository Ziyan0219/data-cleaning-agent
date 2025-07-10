"""
FastAPI Web Interface for Data Cleaning Agent System

This module provides a web-based interface for the data cleaning agent system.
"""

import asyncio
import os
import tempfile
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.agents.main_controller import process_cleaning_request
from src.config.settings import get_settings
from src.utils.json_utils import convert_numpy_types, NumpyJSONEncoder

# Initialize FastAPI app
app = FastAPI(
    title="Data Cleaning Agent System",
    description="Intelligent data cleaning system based on LangChain and LangGraph",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create static directory if it doesn't exist
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Pydantic models
class CleaningRequest(BaseModel):
    requirements: str
    data_content: Optional[str] = None

class CleaningResponse(BaseModel):
    session_id: str
    status: str
    execution_time: float
    quality_metrics: Dict[str, Any]
    results: Dict[str, Any]
    error: Optional[str] = None

# Global storage for results (in production, use a database)
cleaning_results = {}

@app.get("/", response_class=HTMLResponse)
async def home():
    """Home page with web interface"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Data Cleaning Agent System</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #333; text-align: center; margin-bottom: 30px; }
            .form-group { margin-bottom: 20px; }
            label { display: block; margin-bottom: 5px; font-weight: bold; color: #555; }
            textarea, input[type="file"] { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 5px; font-size: 14px; }
            textarea { height: 120px; resize: vertical; }
            button { background-color: #007bff; color: white; padding: 12px 30px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; margin-right: 10px; }
            button:hover { background-color: #0056b3; }
            button:disabled { background-color: #ccc; cursor: not-allowed; }
            .results { margin-top: 30px; padding: 20px; background-color: #f8f9fa; border-radius: 5px; border-left: 4px solid #007bff; }
            .error { border-left-color: #dc3545; background-color: #f8d7da; }
            .success { border-left-color: #28a745; background-color: #d4edda; }
            .loading { text-align: center; color: #007bff; }
            .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-top: 15px; }
            .metric { background: white; padding: 15px; border-radius: 5px; border: 1px solid #e9ecef; }
            .metric-value { font-size: 24px; font-weight: bold; color: #007bff; }
            .metric-label { color: #6c757d; font-size: 14px; }
            pre { background: #f8f9fa; padding: 15px; border-radius: 5px; overflow-x: auto; white-space: pre-wrap; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 Data Cleaning Agent System</h1>
            
            <form id="cleaningForm">
                <div class="form-group">
                    <label for="requirements">Cleaning Requirements:</label>
                    <textarea id="requirements" name="requirements" placeholder="Describe your data cleaning requirements, e.g., 'Handle missing values, remove duplicates, standardize formats'..." required></textarea>
                </div>
                
                <div class="form-group">
                    <label for="dataFile">Upload Data File (CSV, Excel, JSON):</label>
                    <input type="file" id="dataFile" name="dataFile" accept=".csv,.xlsx,.xls,.json" required>
                </div>
                
                <button type="submit" id="submitBtn">🚀 Start Cleaning</button>
                <button type="button" id="clearBtn" onclick="clearResults()">🗑️ Clear Results</button>
            </form>
            
            <div id="results" style="display: none;"></div>
        </div>

        <script>
            document.getElementById('cleaningForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const submitBtn = document.getElementById('submitBtn');
                const resultsDiv = document.getElementById('results');
                
                // Show loading
                submitBtn.disabled = true;
                submitBtn.textContent = '⏳ Processing...';
                resultsDiv.style.display = 'block';
                resultsDiv.innerHTML = '<div class="results loading"><h3>🔄 Processing your data...</h3><p>This may take a few moments. Please wait.</p></div>';
                
                try {
                    const formData = new FormData();
                    formData.append('requirements', document.getElementById('requirements').value);
                    formData.append('file', document.getElementById('dataFile').files[0]);
                    
                    const response = await fetch('/clean', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    
                    if (result.status === 'completed') {
                        displaySuccess(result);
                    } else {
                        displayError(result.error || 'Unknown error occurred');
                    }
                    
                } catch (error) {
                    displayError('Network error: ' + error.message);
                } finally {
                    submitBtn.disabled = false;
                    submitBtn.textContent = '🚀 Start Cleaning';
                }
            });
            
            function displaySuccess(result) {
                const html = `
                    <div class="results success">
                        <h3>✅ Cleaning Completed Successfully!</h3>
                        <p><strong>Session ID:</strong> ${result.session_id}</p>
                        <p><strong>Execution Time:</strong> ${result.execution_time.toFixed(2)} seconds</p>
                        
                        <h4>📊 Quality Metrics:</h4>
                        <div class="metrics">
                            ${Object.entries(result.quality_metrics || {}).map(([key, value]) => `
                                <div class="metric">
                                    <div class="metric-value">${typeof value === 'number' ? value.toFixed(2) : value}</div>
                                    <div class="metric-label">${key.replace('_', ' ').toUpperCase()}</div>
                                </div>
                            `).join('')}
                        </div>
                        
                        <h4>📋 Processing Summary:</h4>
                        <pre>${JSON.stringify(result.results, null, 2)}</pre>
                        
                        <div style="margin-top: 20px;">
                            <button onclick="downloadResults('${result.session_id}')">💾 Download Cleaned Data</button>
                            <button onclick="downloadReport('${result.session_id}')">📄 Download Report</button>
                        </div>
                    </div>
                `;
                document.getElementById('results').innerHTML = html;
            }
            
            function displayError(error) {
                const html = `
                    <div class="results error">
                        <h3>❌ Cleaning Failed</h3>
                        <p><strong>Error:</strong> ${error}</p>
                        <p>Please check your data format and requirements, then try again.</p>
                    </div>
                `;
                document.getElementById('results').innerHTML = html;
            }
            
            function clearResults() {
                document.getElementById('results').style.display = 'none';
                document.getElementById('cleaningForm').reset();
            }
            
            async function downloadResults(sessionId) {
                window.open(`/download/data/${sessionId}`, '_blank');
            }
            
            async function downloadReport(sessionId) {
                window.open(`/download/report/${sessionId}`, '_blank');
            }
        </script>
    </body>
    </html>
    """
    return html_content

@app.post("/clean", response_model=CleaningResponse)
async def clean_data(
    requirements: str = Form(...),
    file: UploadFile = File(...)
):
    """Clean data endpoint"""
    try:
        # Validate file type
        allowed_types = ['.csv', '.xlsx', '.xls', '.json']
        file_extension = Path(file.filename).suffix.lower()
        
        if file_extension not in allowed_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type. Allowed types: {', '.join(allowed_types)}"
            )
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Process cleaning request
            result = await process_cleaning_request(
                user_requirements=requirements,
                data_source=temp_file_path
            )
            
            # Store result for later download
            session_id = result['session_id']
            cleaning_results[session_id] = result
            
            # Convert numpy types for JSON serialization
            safe_result = convert_numpy_types(result)
            
            return CleaningResponse(
                session_id=session_id,
                status=safe_result['status'],
                execution_time=safe_result.get('execution_time', 0),
                quality_metrics=safe_result.get('quality_metrics', {}),
                results=safe_result.get('results', {}),
                error=safe_result.get('error')
            )
            
        finally:
            # Clean up temporary file
            os.unlink(temp_file_path)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/data/{session_id}")
async def download_cleaned_data(session_id: str):
    """Download cleaned data"""
    if session_id not in cleaning_results:
        raise HTTPException(status_code=404, detail="Session not found")
    
    result = cleaning_results[session_id]
    
    if result['status'] != 'completed' or not result.get('final_data'):
        raise HTTPException(status_code=400, detail="No cleaned data available")
    
    # Create temporary file with cleaned data
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as temp_file:
        temp_file.write(result['final_data'])
        temp_file_path = temp_file.name
    
    return FileResponse(
        temp_file_path,
        media_type='text/csv',
        filename=f"cleaned_data_{session_id}.csv"
    )

@app.get("/download/report/{session_id}")
async def download_report(session_id: str):
    """Download cleaning report"""
    if session_id not in cleaning_results:
        raise HTTPException(status_code=404, detail="Session not found")
    
    result = cleaning_results[session_id]
    
    # Generate report
    from src.agents.result_aggregation_agent import ResultAggregationAgent
    from langchain_openai import ChatOpenAI
    
    settings = get_settings()
    llm = ChatOpenAI(
        model=settings.llm.model,
        api_key=settings.llm.api_key,
        temperature=0
    )
    
    aggregation_agent = ResultAggregationAgent(llm)
    report = aggregation_agent.generate_report(result.get('results', {}))
    
    # Create temporary file with report
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.md') as temp_file:
        temp_file.write(report)
        temp_file_path = temp_file.name
    
    return FileResponse(
        temp_file_path,
        media_type='text/markdown',
        filename=f"cleaning_report_{session_id}.md"
    )

@app.get("/api/status")
async def get_status():
    """Get system status"""
    settings = get_settings()
    
    return {
        "status": "running",
        "version": "1.0.0",
        "model": settings.llm.model,
        "active_sessions": len(cleaning_results)
    }

@app.get("/api/sessions")
async def get_sessions():
    """Get all cleaning sessions"""
    sessions = []
    for session_id, result in cleaning_results.items():
        sessions.append({
            "session_id": session_id,
            "status": result['status'],
            "execution_time": result.get('execution_time', 0),
            "timestamp": result.get('timestamp', 'unknown')
        })
    
    return {"sessions": sessions}

if __name__ == "__main__":
    import uvicorn
    
    # Check configuration
    settings = get_settings()
    if not settings.validate_config():
        print("❌ Configuration validation failed")
        exit(1)
    
    print("🚀 Starting Data Cleaning Agent Web Interface...")
    print("📱 Open your browser and go to: http://localhost:8000")
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

