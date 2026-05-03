@echo off
echo ===================================================
echo BIS Standards Recommendation Engine - Startup Script
echo ===================================================

echo.
echo [1] Checking dependencies...
pip install -r requirements.txt -q

echo.
echo [2] Checking Vector Database...
if not exist "chromadb" (
    echo Vector database not found. Building from dataset.pdf...
    python build_vectorstore.py
    if %ERRORLEVEL% neq 0 (
        echo Error building vector database.
        pause
        exit /b %ERRORLEVEL%
    )
) else (
    echo Vector database found.
)

echo.
echo [3] Generating Inference Results for Hackathon Submission...
python inference.py --input data/public_test_set.json --output results.json
if %ERRORLEVEL% neq 0 (
    echo Error running inference.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [4] Running Evaluation Script against Hackathon Criteria...
python eval_script.py --results results.json

echo.
echo ===================================================
echo Starting Web UI for Demonstration...
echo Press Ctrl+C to exit
echo ===================================================
python app.py

pause
