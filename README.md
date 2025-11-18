# PR_project

A comprehensive machine learning project combining reinforcement learning agents with rule-based expert systems for financial market analysis. This project integrates order book data processing, feature engineering, RL training, and explainable AI components with an interactive dashboard for visualization and monitoring.

## Quick Start

1. **Bootstrap the project structure:**
   ```bash
   chmod +x bootstrap_project.sh
   ./bootstrap_project.sh
   ```

2. **Set up virtual environment and install dependencies:**
   ```bash
   chmod +x scripts/setup_env.sh
   ./scripts/setup_env.sh
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Initialize git repository:**
   ```bash
   chmod +x scripts/init_repo.sh
   ./scripts/init_repo.sh
   ```

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

## Next Steps

1. **Data Ingestion** - Add your market data to `data/raw/`
2. **Preprocessing** - Implement data cleaning and feature extraction
3. **RL Agent Training** - Configure and train reinforcement learning models
4. **Rule Integration** - Develop expert rules and fusion strategies
5. **Dashboard Development** - Build interactive monitoring interface