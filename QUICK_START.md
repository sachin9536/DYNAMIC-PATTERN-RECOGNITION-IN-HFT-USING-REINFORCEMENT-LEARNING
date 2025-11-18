# 🚀 Quick Start Guide - Market Anomaly Detection System

## ✅ System Status: READY TO RUN

The system has been successfully implemented and tested. All components are working correctly.

## 🎯 Fastest Way to Get Started

### Option 1: Dashboard Only (Recommended for First Time)

```bash
# 1. Navigate to project directory
cd PR_project

# 2. Activate virtual environment (if not already active)
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# 3. Start the dashboard
streamlit run src/dashboard/main_app.py
```

**Then open:** http://localhost:8501

### Option 2: Full System (API + Dashboard)

#### Terminal 1 - Start API:
```bash
cd PR_project
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/macOS

# Start API server
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

#### Terminal 2 - Start Dashboard:
```bash
cd PR_project
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/macOS

# Start dashboard
streamlit run src/dashboard/main_app.py
```

**Access Points:**
- Dashboard: http://localhost:8501
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs

## 🎮 What You Can Do Right Now

### 1. Dashboard Features Available:

#### 📊 **Overview Page**
- System health monitoring
- Key performance metrics
- Quick action buttons

#### 🎯 **Live Simulation**
- Click "📊 Demo" to run market simulation
- Watch real-time charts update
- See performance metrics

#### 🤖 **Model Monitor**
- View available models (initially empty)
- Model management interface
- Performance tracking

#### 🔍 **Explainability**
- Click "🎲 Generate Sample Data"
- Select explanation method (Rule-based works immediately)
- Click "🔍 Generate Explanation"
- View feature importance and explanations

#### 📋 **Rules Audit**
- Click "🎲 Generate Sample Audit Data"
- View rule trigger logs
- Analyze rule performance
- Test rule system with custom data

#### 📈 **Training Monitor**
- Click "🎲 Generate Sample Training Data"
- View training run history
- Monitor training progress
- TensorBoard integration info

### 2. API Features Available:

Visit http://localhost:8000/docs to explore:
- `/health` - System health check
- `/models` - List available models
- `/predict` - Make predictions (requires model)
- `/rules/check` - Test rule system
- `/metrics` - Prometheus metrics

## 🧪 Test the System

Run the comprehensive system test:
```bash
python test_system.py
```

Expected output: `🎉 ALL TESTS PASSED! System is ready to run.`

## 📚 Key Files and Directories

```
PR_project/
├── src/
│   ├── api/           # FastAPI REST API
│   ├── dashboard/     # Streamlit dashboard
│   ├── explainability/  # SHAP/LIME/Rules
│   ├── agents/        # RL training
│   └── data/          # Data processing
├── configs/           # Configuration files
├── artifacts/         # Models, logs, results
├── scripts/           # Utility scripts
└── RUN_GUIDE.md      # Detailed run instructions
```

## 🔧 Configuration

The system is pre-configured and ready to run. Key settings in `configs/config.yaml`:

- **Dashboard:** Port 8501, auto-refresh enabled
- **API:** Port 8000, CORS enabled
- **Monitoring:** Prometheus metrics, structured logging
- **Simulation:** Real-time updates, demo data generation

## 🎨 Sample Data

The system generates realistic sample data automatically:
- **Market Data:** Synthetic price/volume data with anomalies
- **Training Runs:** Sample training history and metrics
- **Rule Logs:** Sample rule trigger events
- **Explanations:** Feature importance and rule explanations

## 🚨 Troubleshooting

### If Dashboard Won't Start:
```bash
# Check if port is in use
netstat -an | findstr :8501  # Windows
# netstat -tulpn | grep :8501  # Linux

# Try different port
streamlit run src/dashboard/main_app.py --server.port 8502
```

### If API Won't Start:
```bash
# Check if port is in use
netstat -an | findstr :8000  # Windows

# Try different port
uvicorn src.api.app:app --port 8001
```

### If Imports Fail:
```bash
# Set Python path
set PYTHONPATH=%PYTHONPATH%;%CD%\src  # Windows
# export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"  # Linux/macOS
```

## 🎯 Next Steps

1. **Explore the Dashboard:** Navigate through all pages and try the demo features
2. **Generate Your Data:** Replace sample data with real market data
3. **Train Models:** Use the training scripts to create your own models
4. **Customize Rules:** Modify the rule system for your specific needs
5. **Set Up Monitoring:** Configure Prometheus/Grafana for production

## 📞 Need Help?

1. Check `RUN_GUIDE.md` for detailed instructions
2. Run `python test_system.py` to verify system health
3. Check console output for error messages
4. Review configuration in `configs/config.yaml`

---

**🎉 Enjoy exploring your Market Anomaly Detection System!**