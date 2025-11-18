# System Commands Reference

## Quick Start Commands

### Windows (PowerShell)
```powershell
# Start Dashboard
streamlit run src/dashboard/main_app.py --server.port 8502

# Start API
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000

# Train Model
.\scripts\run_train.ps1

# Evaluate Model
python -m src.agents.evaluate --model artifacts/final_model.zip --config configs/config.yaml --n_episodes 20
```

### Linux/macOS (Bash)
```bash
# Start Dashboard
streamlit run src/dashboard/main_app.py

# Start API
uvicorn src.api.app:app --host 0.0.0.0 --port 8000

# Train Model
./scripts/run_train.sh

# Evaluate Model
python -m src.agents.evaluate --model artifacts/final_model.zip --config configs/config.yaml --n_episodes 20
```

## System URLs
- Dashboard: http://localhost:8502
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- API Health: http://localhost:8000/health
