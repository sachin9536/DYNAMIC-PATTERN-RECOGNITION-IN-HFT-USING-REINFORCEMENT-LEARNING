# PowerShell Training script for Market Anomaly Detection System
Write-Host "Starting RL Agent Training..." -ForegroundColor Green

# Check if we're in the right directory
if (-not (Test-Path "src/agents/train_agent.py")) {
    Write-Host "Error: src/agents/train_agent.py not found. Please run from project root." -ForegroundColor Red
    exit 1
}

# Set environment variables
$env:PYTHONPATH = "$env:PYTHONPATH;$(Get-Location)\src"

# Default configuration
$CONFIG_FILE = if ($env:CONFIG_FILE) { $env:CONFIG_FILE } else { "configs/config.yaml" }
$ALGORITHM = if ($env:ALGORITHM) { $env:ALGORITHM } else { "ppo" }
$TIMESTEPS = if ($env:TIMESTEPS) { $env:TIMESTEPS } else { "2000" }
$OUTPUT_DIR = if ($env:OUTPUT_DIR) { $env:OUTPUT_DIR } else { "artifacts/" }

Write-Host "Training Configuration:" -ForegroundColor Cyan
Write-Host "  Config File: $CONFIG_FILE" -ForegroundColor White
Write-Host "  Algorithm: $ALGORITHM" -ForegroundColor White
Write-Host "  Timesteps: $TIMESTEPS" -ForegroundColor White
Write-Host "  Output Directory: $OUTPUT_DIR" -ForegroundColor White

# Check if config file exists
if (-not (Test-Path $CONFIG_FILE)) {
    Write-Host "Error: Config file not found: $CONFIG_FILE" -ForegroundColor Red
    exit 1
}

# Create output directory if it doesn't exist
if (-not (Test-Path $OUTPUT_DIR)) {
    New-Item -ItemType Directory -Path $OUTPUT_DIR -Force | Out-Null
}

# Run training
Write-Host "Starting training with $ALGORITHM algorithm..." -ForegroundColor Yellow
python -m src.agents.train_agent --config $CONFIG_FILE --algo $ALGORITHM --timesteps $TIMESTEPS --out_dir $OUTPUT_DIR

if ($LASTEXITCODE -eq 0) {
    Write-Host "Training completed! Check $OUTPUT_DIR for results." -ForegroundColor Green
} else {
    Write-Host "Training failed with exit code $LASTEXITCODE" -ForegroundColor Red
}