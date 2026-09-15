Write-Host "Starting FastAPI Backend on port 8000..." -ForegroundColor Green
Start-Process python -ArgumentList "-m uvicorn app.main:app --port 8000 --reload"

Write-Host "Starting Streamlit Dashboard on port 8501..." -ForegroundColor Green
Start-Process python -ArgumentList "-m streamlit run dashboard.py"

Write-Host "`nBoth services launched!" -ForegroundColor Cyan
Write-Host "API Docs: http://127.0.0.1:8000/docs"
Write-Host "Dashboard: http://localhost:8501"
