"""
FastAPI Web Interface for Data Cleaning Agent - Working Version

This module provides a web interface using the simplified, working controller.
"""

import asyncio
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

from src.agents.main_controller_fixed import process_cleaning_request
from src.utils.json_utils import convert_numpy_types

# Initialize FastAPI app
app = FastAPI(
    title="Cattle Data Cleaning Agent - Working Version",
    description="AI-powered cattle data quality control system",
    version="2.1.0"
)

# Response models
class CleaningResponse(BaseModel):
    status: str
    message: str
    report: str = ""
    download_url: str = ""
    execution_time: float = 0.0
    quality_score: float = 0.0

@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the main web interface."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>🐄 Cattle Data Cleaning Agent</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #2c3e50; text-align: center; margin-bottom: 30px; }
            .form-group { margin-bottom: 20px; }
            label { display: block; margin-bottom: 5px; font-weight: 600; color: #34495e; }
            input, textarea, select { width: 100%; padding: 12px; border: 2px solid #ddd; border-radius: 5px; font-size: 16px; }
            input:focus, textarea:focus { border-color: #3498db; outline: none; }
            button { background: #3498db; color: white; padding: 15px 30px; border: none; border-radius: 5px; font-size: 16px; cursor: pointer; width: 100%; }
            button:hover { background: #2980b9; }
            button:disabled { background: #bdc3c7; cursor: not-allowed; }
            .result { margin-top: 30px; padding: 20px; background: #f8f9fa; border-radius: 5px; border-left: 4px solid #3498db; }
            .error { border-left-color: #e74c3c; background: #fdf2f2; }
            .success { border-left-color: #27ae60; background: #f2fdf2; }
            .loading { text-align: center; color: #7f8c8d; }
            .report { white-space: pre-wrap; font-family: 'Courier New', monospace; background: white; padding: 15px; border-radius: 5px; margin-top: 15px; }
            .download-btn { background: #27ae60; margin-top: 15px; display: inline-block; text-decoration: none; color: white; padding: 10px 20px; border-radius: 5px; }
            .stats { display: flex; justify-content: space-around; margin: 20px 0; }
            .stat { text-align: center; }
            .stat-value { font-size: 24px; font-weight: bold; color: #3498db; }
            .stat-label { color: #7f8c8d; font-size: 14px; }
            .version-info { text-align: center; color: #7f8c8d; font-size: 12px; margin-top: 20px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🐄 Cattle Data Cleaning Agent</h1>
            <p style="text-align: center; color: #7f8c8d; margin-bottom: 30px;">
                AI-powered quality control for cattle weight data and ready-to-load status
            </p>
            
            <form id="cleaningForm" enctype="multipart/form-data">
                <div class="form-group">
                    <label for="file">📁 Upload Cattle Data (CSV)</label>
                    <input type="file" id="file" name="file" accept=".csv" required>
                    <small style="color: #7f8c8d;">Upload your cattle lot data with weight and status information</small>
                </div>
                
                <div class="form-group">
                    <label for="requirements">📋 Cleaning Requirements</label>
                    <textarea id="requirements" name="requirements" rows="3" placeholder="e.g., Process cattle data for weight validation and label correction">Process cattle data for weight validation and label correction</textarea>
                </div>
                
                <button type="submit" id="submitBtn">🚀 Start Data Cleaning</button>
            </form>
            
            <div id="result" style="display: none;"></div>
            
            <div class="version-info">
                Version 2.1.0 - Simplified & Reliable | Working Controller
            </div>
        </div>
        
        <script>
            document.getElementById('cleaningForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const submitBtn = document.getElementById('submitBtn');
                const resultDiv = document.getElementById('result');
                
                // Show loading state
                submitBtn.disabled = true;
                submitBtn.textContent = '🔄 Processing...';
                resultDiv.style.display = 'block';
                resultDiv.className = 'result loading';
                resultDiv.innerHTML = '<h3>🔄 Processing your cattle data...</h3><p>This may take a few moments. Please wait.</p>';
                
                try {
                    const formData = new FormData();
                    formData.append('file', document.getElementById('file').files[0]);
                    formData.append('requirements', document.getElementById('requirements').value);
                    
                    const response = await fetch('/clean', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    
                    if (result.status === 'completed') {
                        resultDiv.className = 'result success';
                        resultDiv.innerHTML = `
                            <h3>✅ Cleaning Completed Successfully!</h3>
                            <div class="stats">
                                <div class="stat">
                                    <div class="stat-value">${result.execution_time.toFixed(1)}s</div>
                                    <div class="stat-label">Processing Time</div>
                                </div>
                                <div class="stat">
                                    <div class="stat-value">${result.quality_score.toFixed(1)}%</div>
                                    <div class="stat-label">Quality Score</div>
                                </div>
                            </div>
                            <div class="report">${result.report}</div>
                            ${result.download_url ? `<a href="${result.download_url}" class="download-btn">📥 Download Cleaned Data</a>` : ''}
                        `;
                    } else {
                        resultDiv.className = 'result error';
                        resultDiv.innerHTML = `
                            <h3>❌ Cleaning Failed</h3>
                            <p><strong>Error:</strong> ${result.message}</p>
                            <p>Please check your data format and requirements, then try again.</p>
                        `;
                    }
                } catch (error) {
                    resultDiv.className = 'result error';
                    resultDiv.innerHTML = `
                        <h3>❌ Cleaning Failed</h3>
                        <p><strong>Error:</strong> ${error.message}</p>
                        <p>Please check your data format and requirements, then try again.</p>
                    `;
                }
                
                // Reset button
                submitBtn.disabled = false;
                submitBtn.textContent = '🚀 Start Data Cleaning';
            });
        </script>
    </body>
    </html>
    """

@app.post("/clean")
async def clean_data(
    file: UploadFile = File(...),
    requirements: str = Form(...)
):
    """Process uploaded cattle data file."""
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.csv', delete=False) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Process the data using the fixed controller
            result = await process_cleaning_request(
                user_requirements=requirements,
                data_source=temp_file_path
            )
            
            # Convert numpy types for JSON serialization
            safe_result = convert_numpy_types(result)
            
            # Save cleaned data if available
            download_url = ""
            if safe_result.get("status") == "completed" and safe_result.get("cleaned_data"):
                # Save cleaned data to output directory
                output_dir = Path("data/output")
                output_dir.mkdir(exist_ok=True)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                cleaned_file_path = output_dir / f"cleaned_cattle_data_{timestamp}.csv"
                
                with open(cleaned_file_path, 'w', encoding='utf-8') as f:
                    f.write(safe_result["cleaned_data"])
                
                download_url = f"/download/{cleaned_file_path.name}"
            
            # Prepare response
            if safe_result.get("status") == "completed":
                return CleaningResponse(
                    status="completed",
                    message="Data cleaning completed successfully",
                    report=safe_result.get("final_report", "Report generation failed"),
                    download_url=download_url,
                    execution_time=safe_result.get("execution_time", 0.0),
                    quality_score=safe_result.get("quality_score", 0.0)
                )
            else:
                return CleaningResponse(
                    status="error",
                    message=safe_result.get("error", "Unknown error occurred")
                )
                
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
            except:
                pass
                
    except Exception as e:
        return CleaningResponse(
            status="error",
            message=f"Processing failed: {str(e)}"
        )

@app.get("/download/{filename}")
async def download_file(filename: str):
    """Download cleaned data file."""
    file_path = Path("data/output") / filename
    if file_path.exists():
        return FileResponse(
            path=str(file_path),
            filename=filename,
            media_type='text/csv'
        )
    else:
        raise HTTPException(status_code=404, detail="File not found")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/test")
async def test_endpoint():
    """Test endpoint to verify the system is working."""
    try:
        # Test with sample data
        sample_data_path = "data/input/cattle_data.csv"
        if Path(sample_data_path).exists():
            result = await process_cleaning_request(
                user_requirements="Test processing",
                data_source=sample_data_path
            )
            return {"status": "test_passed", "result_status": result.get("status")}
        else:
            return {"status": "test_failed", "error": "Sample data not found"}
    except Exception as e:
        return {"status": "test_failed", "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

