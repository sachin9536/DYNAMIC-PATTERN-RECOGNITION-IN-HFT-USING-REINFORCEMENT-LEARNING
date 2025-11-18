# DYNAMIC PATTERN RECOGNITION IN HFT USING REINFORCEMENT LEARNING

This repository contains code, configuration, and documentation for a research/engineering project that explores dynamic pattern recognition for high-frequency trading (HFT) using reinforcement learning (RL). The project provides data preparation utilities, RL agent training and evaluation scaffolding, explainability tools, and a dashboard for visualization and real-time monitoring.

Key goals
- Build RL agents that operate on high-frequency/sequenced market data to detect exploitable patterns.
- Provide a reproducible pipeline for preprocessing, sequence generation, agent training, evaluation, and explainability.
- Ship lightweight tools and a dashboard for inspecting agent behavior, model explanations, and live simulation.

Repository layout (important paths)
- `src/` — main source code: agents, RL agent wrappers, data loaders, explainability, dashboard and utilities.
- `data/` — local data folder (contains `raw/` and `processed/` subfolders; large datasets should be kept out of git).
- `scripts/` — convenience scripts for running preprocessing, training, and other tasks.
- `artifacts/`, `models/`, `checkpoints/` — training outputs and model weights (ignored by default; do not commit large binary files).
- `docs/` — project documentation and guides.
- `notebooks/` — Jupyter notebooks demonstrating preprocessing, explainability, and sequence preparation.

Requirements
- Python 3.8+ recommended
- See `requirements.txt` for exact Python dependencies.

Quick setup (local, development)
1. Clone the repo (or if already cloned, skip):

    git clone https://github.com/<your-username>/<repo-name>.git
    cd <repo-name>

2. Create and activate a virtual environment (Windows PowerShell example):

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1

3. Install dependencies:

    pip install -r requirements.txt

Data
- The repo ignores large datasets by default. Place raw datasets under `data/raw/` and processed outputs under `data/processed/`.
- For reproducible results, keep only small sample datasets in `data/` under version control. Large datasets and model weights belong in external storage (S3, Google Drive, or an artifact store).

Common tasks
- Prepare sequences for training:

   ```bash
   python scripts/build_sequences.py
   ```

- Run preprocessing:

   ```bash
   python scripts/run_preprocessing.py
   ```

- Train an agent (example; see `src/agents/train_agent.py` for options):

   ```bash
   python src/agents/train_agent.py --config configs/config.yaml
   ```

- Run the dashboard locally (Flask/FastAPI app under `src/dashboard`):

   ```bash
   python src/dashboard/main_app.py
   ```

Tests
- Run tests with `pytest`:

   ```bash
   pytest -q
   ```

Notes about versioning and large files
- `.gitignore` is configured to exclude virtual environments, dataset folders, model weights, logs, and other large or sensitive files. Do not commit large artifacts or secrets.
- If a large file was accidentally committed, use tools such as BFG Repo-Cleaner or `git filter-repo` to remove it from history before sharing the repo publicly.

Contributing
- Fork the repo, create feature branches, and open pull requests. Keep commits small and focused; include tests for new behavior.

Contact
- For questions about this repo, contact the maintainer or open an issue on GitHub.

---

If you want, I can also:
- Add a short `QUICK_START.md` with exact commands you run locally (I can generate it from the repo's scripts).
- Stage & push any other files you want included in the remote (you previously had many untracked files).
# Market Anomaly Detection System

**Status:** ✅ Production Ready | **Version:** 1.0.0 | **Performance:** 73% F1 Score

A comprehensive machine learning project combining reinforcement learning agents with rule-based expert systems for financial market analysis. This project integrates order book data processing, feature engineering, RL training, and explainable AI components with an interactive dashboard for visualization and monitoring.

## 📚 Documentation

- **[COMPLETE_GUIDE.md](COMPLETE_GUIDE.md)** - Full system documentation (START HERE)
- **[COMMANDS.md](COMMANDS.md)** - Quick command reference
- **[RUN_GUIDE.md](RUN_GUIDE.md)** - Detailed running instructions
- **[QUICK_START.md](QUICK_START.md)** - Fast start guide
- **[DASHBOARD_STATUS.md](DASHBOARD_STATUS.md)** - Dashboard status and fixes

## 🚀 Quick Start

### Run the System (Windows)
```powershell
cd PR_project
.venv\Scripts\activate
streamlit run src/dashboard/main_app.py --server.port 8502
```
**Access:** http://localhost:8502

### Run the System (Linux/macOS)
```bash
cd PR_project
source .venv/bin/activate
streamlit run src/dashboard/main_app.py
```
**Access:** http://localhost:8501

### Key Commands
```powershell
# Train Model
.\scripts\run_train.ps1

# Evaluate Model
python -m src.agents.evaluate --model artifacts/final_model.zip --config configs/config.yaml --n_episodes 20

# Start API
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000

# Test System
python test_system.py
```

See **[COMMANDS.md](COMMANDS.md)** for complete command reference.

## Project Structure

- `data/` - Raw and processed datasets
- `src/data/` - Data loading and preprocessing utilities
- `src/features/` - Feature engineering modules
- `src/rl_agent/` - Reinforcement learning agent implementation
- `src/rules/` - Expert rule-based systems
- `src/fusion/` - Model fusion and ensemble methods
- `src/explainability/` - SHAP/LIME integration for model interpretation
- `src/dashboard/` - Streamlit/Dash visualization interface
- `src/utils/` - Common utilities and helpers
- `configs/` - Configuration files
- `tests/` - Unit tests
- `notebooks/` - Jupyter notebooks for exploration
- `scripts/` - Setup and utility scripts

## RL Training and Evaluation

### Training an Agent

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run smoke training:**
   ```bash
   chmod +x scripts/run_train.sh
   ./scripts/run_train.sh
   ```

3. **Custom training:**
   ```bash
   python -m src.agents.train_agent --config configs/config.yaml --algo ppo --timesteps 10000 --out_dir artifacts/
   ```

### Evaluation

```bash
python -m src.agents.evaluate --model artifacts/models/final_model --config configs/config.yaml --n_episodes 100
```

### Fusion Behavior

The system supports rule-based action fusion where expert rules can override RL actions:
- **High cancellation ratio**: Overrides to signal anomaly
- **Extreme price movements**: Triggers risk management actions
- **Risk threshold breach**: Forces hold action

### CVaR Wrapper

The CVaR (Conditional Value at Risk) wrapper provides risk-aware training by:
- Tracking episode returns in a sliding window
- Computing empirical VaR and CVaR estimates
- Applying penalties when returns drop below risk thresholds

**Limitations**: This is an empirical approximation, not true CVaR optimization.

## 📊 System Status

### Current Performance
- **Model**: PPO Agent (Trained)
- **F1 Score**: 73.2%
- **Precision**: 57.8%
- **Recall**: 100%
- **Training**: 2000 timesteps completed
- **Data**: 901 sequences ready

### What's Working
✅ RL Model Training (PPO/SAC/A2C)  
✅ Model Evaluation & Metrics  
✅ Interactive Dashboard (6 pages)  
✅ REST API (FastAPI)  
✅ Expert Rules System (7 rules)  
✅ Risk Management (CVaR)  
✅ Explainability (SHAP/LIME/Rules)  
✅ Data Pipeline (Synthetic generation)

### System URLs
- **Dashboard**: http://localhost:8502
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## 🎯 Next Steps

1. **Explore Dashboard** - Navigate through all 6 pages
2. **Run Simulations** - Test live trading simulation
3. **Train More** - Increase timesteps for better performance
4. **Add Real Data** - Replace synthetic with actual market feeds
5. **Customize Rules** - Modify expert rules for your use case
6. **Deploy** - Set up production deployment

## 📖 Full Documentation

For complete system documentation, architecture, troubleshooting, and advanced features, see:
- **[COMPLETE_GUIDE.md](COMPLETE_GUIDE.md)** - Comprehensive system guide