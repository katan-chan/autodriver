@echo off

:: Check if data exists
if not exist "data\hanoi_component_nodes.csv" (
    echo [INFO] Data missing. Running export script to generate graph cache...
    
    :: Install dependencies first to ensure script runs
    cd backend
    pip install -r requirements.txt
    cd ..
    
    :: Run generation script
    python src/export_hanoi_component.py
    echo [INFO] Data generation complete.
) else (
    echo [INFO] Data found. Skipping generation.
)

echo Starting Backend...
start "Backend" cmd /k "cd backend && pip install -r requirements.txt && python -m app.main"

echo Starting Frontend...
start "Frontend" cmd /k "cd frontend && npm install && npm run dev"

echo Process started!
