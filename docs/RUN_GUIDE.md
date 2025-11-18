# 🚀 Market Anomaly Detection System - Run Guide

This comprehensive guide explains how to run all components of the Market Anomaly Detection System.

## 📋 Prerequisites

### System Requirements
- Python 3.11 or higher
- 4GB+ RAM (8GB+ recommended)
- 2GB+ free disk space
- Windows/Linux/macOS

### Required Dependencies
All dependencies are listed in `requirements.txt` and will be installed automatically.

## 🛠️ Initial Setup

### 1. Environment Setup
```bash
# Navigate to project directory
cd PR_project

# Make scripts executable (Linux/macOS)
chmod +x scripts/*.sh

# Set up virtual environment and install dependencies
./scripts/setup_env.sh

# Activate virtual environment
# Linux/macOS:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate
```

### 2. Verify Installation
```bash
# Run setup verification
python verify_setup.py

# Run milestone verifications
python verify_milestone2.py  # Data pipeline
python verify_milestone3.py  # Feature engineering
python verify_milestone4.py  # RL environment
python verify_milestone5.py  # Explainability
python verify_milestone6.py  # Dashboard
```

## 🎯 Running the Application

The system has multiple components that can be run independently or together:

### Option 1: Quick Start - Dashboard Only (Recommended for First Run)

```bash
# Start the Streamlit dashboard
streamlit run src/dashboard/main_app.py

# Or using Python module
python -m streamlit run src/dashboard/main_app.py --server.port 8501
```

**Access:** http://localhost:8501

**Features Available:**
- System overview and health monitoring
- Live market simulation with demo data
- Model management (if models are available)
- Explainability analysis with sample data
- Rules audit with generated sample data
- Training monitoring with sample data

### Option 2: Full System - API + Dashboard

#### Terminal 1: Start the API Server
```bash
# Using the provided script
./scripts/run_api.sh

# Or manually
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

**Access:** 
- API: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

#### Terminal 2: Start the Dashboard
```bash
streamlit run src/dashboard/main_app.py
```

**Access:** http://localhost:8501

**Configure Dashboard for API Mode:**
1. In the dashboard sidebar, select "API" mode
2. Set API URL to `http://localhost:8000`
3. Test the connection using the "Test API Connection" button

### Option 3: Individual Components

#### A. Data Processing
```bash
# Run data preprocessing
python scripts/run_preprocessing.py

# Build sequences for training
python scripts/build_sequences.py
```

#### B. Model Training
```bash
# Quick training (2000 timesteps)
./scripts/run_train.sh

# Custom training
python -m src.agents.train_agent \
    --config configs/config.yaml \
    --algo ppo \
    --timesteps 10000 \
    --out_dir artifacts/
```

#### C. Model Evaluation
```bash
# Evaluate trained model
python -m src.agents.evaluate \
    --model artifacts/final_model.zip \
    --config configs/config.yaml \
    --n_episodes 100
```

#### D. Explainability Analysis
```bash
# Run explainability tests
python scripts/run_explainability_tests.py

# Or use the notebook
jupyter notebook notebooks/explainability_demo.ipynb
```

## 🔧 Configuration

### Environment Variables
```bash
# API Configuration
export API_HOST=0.0.0.0
export API_PORT=8000
export API_WORKERS=1
export API_RELOAD=true

# Dashboard Configuration  
export DASHBOARD_HOST=0.0.0.0
export DASHBOARD_PORT=8501

# Logging
export LOG_LEVEL=INFO
```

### Configuration File
Edit `configs/config.yaml` to customize:
- Model training parameters
- Data processing settings
- Dashboard appearance
- API behavior
- Monitoring settings

## 📊 Available Features

### 1. Dashboard Pages

#### Overview Page
- System health indicators
- Key performance metrics
- Recent activity logs
- Quick action buttons

#### Live Simulation
- Real-time market data simulation
- Policy evaluation with trained models
- Performance metrics tracking
- Demo mode with random policy

#### Model Monitor
- List and manage trained models
- Model performance comparison
- Loading/unloading models
- Model metadata viewing

#### Explainability
- SHAP explanations (if shap is installed)
- LIME explanations (if lime is installed)
- Rule-based explanations (always available)
- Feature importance analysis

#### Rules Audit
- Rule trigger logs and filtering
- Rule performance analysis
- Rule configuration interface
- Audit log export

#### Training Monitor
- Training run history
- Real-time training progress
- TensorBoard integration
- Training logs viewing

### 2. API Endpoints

#### Health & Status
- `GET /health` - Basic health check
- `GET /status` - Detailed system status

#### Model Management
- `GET /models` - List available models
- `POST /models/{model_id}/load` - Load a model
- `DELETE /models/{model_id}/unload` - Unload a model

#### Inference
- `POST /predict` - Single prediction
- `POST /batch_predict` - Batch predictions

#### Rules
- `POST /rules/check` - Check rules against data
- `GET /rules/summary` - Get rules summary

#### Monitoring
- `GET /metrics` - Prometheus metrics

## 🧪 Testing and Development

### Running Tests
```bash
# Run all tests
pytest

# Run specific test files
pytest tests/test_smoke.py
pytest tests/test_data_pipeline.py
pytest tests/test_env_and_agents.py
```

### Development Mode
```bash
# API with auto-reload
export API_RELOAD=true
./scripts/run_api.sh

# Dashboard with auto-refresh
# (Auto-refresh can be enabled in the dashboard sidebar)
```

### Sample Data Generation
The system can generate synthetic data for testing:

1. **Via Dashboard:** Use "Generate Sample Data" buttons in various pages
2. **Via API:** The system automatically generates demo data when real data is not available
3. **Via Scripts:** Run data generation scripts in the `scripts/` directory

## 🔍 Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# Ensure PYTHONPATH is set
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Or add to your shell profile
echo 'export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"' >> ~/.bashrc
```

#### 2. Port Already in Use
```bash
# Check what's using the port
netstat -tulpn | grep :8000  # Linux
netstat -an | findstr :8000  # Windows

# Use different ports
export API_PORT=8001
export DASHBOARD_PORT=8502
```

#### 3. Missing Dependencies
```bash
# Reinstall requirements
pip install -r requirements.txt

# Install specific missing packages
pip install fastapi uvicorn streamlit plotly
```

#### 4. Model Loading Issues
- Ensure model files are in `artifacts/models/`
- Check model file permissions
- Verify model format compatibility

#### 5. Dashboard Connection Issues
- Verify API is running on the correct port
- Check firewall settings
- Ensure API URL is correctly set in dashboard

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run with verbose output
python -m src.api.app --log-level debug
```

## 📈 Performance Tips

### For Better Performance:
1. **Use API Mode:** Run API and Dashboard separately for better resource management
2. **Limit Data Points:** Reduce `max_data_points` in config for faster loading
3. **Disable Auto-refresh:** Turn off auto-refresh in dashboard for large datasets
4. **Use Caching:** Enable caching in configuration
5. **Close Unused Pages:** Navigate away from resource-intensive pages when not needed

### Resource Usage:
- **API Server:** ~200-500MB RAM
- **Dashboard:** ~300-800MB RAM  
- **Model Training:** 1-4GB RAM (depending on model size)
- **Full System:** 2-6GB RAM total

## 🔗 Useful URLs

When running locally:
- **Dashboard:** http://localhost:8501
- **API:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs
- **API Health:** http://localhost:8000/health
- **Metrics:** http://localhost:8000/metrics

## 📚 Next Steps

1. **Explore the Dashboard:** Start with the Overview page and navigate through all sections
2. **Try Live Simulation:** Use the demo mode to see the system in action
3. **Load Your Data:** Replace sample data with your own market data
4. **Train Models:** Use the training scripts to create your own models
5. **Customize Rules:** Modify the rule system for your specific use case
6. **Set Up Monitoring:** Configure Prometheus/Grafana for production monitoring

## 🆘 Getting Help

1. **Check Logs:** Look at console output for error messages
2. **Verify Setup:** Run the verification scripts
3. **Check Documentation:** Review `docs/` directory for detailed information
4. **Test Components:** Use the individual component tests to isolate issues

---

**Happy Analyzing! 📊🚀**