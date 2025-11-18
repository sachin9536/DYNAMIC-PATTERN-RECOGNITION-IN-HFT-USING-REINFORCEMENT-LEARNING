#!/usr/bin/env bash
set -e

CONFIG=configs/config.yaml
ALGO=ppo
TIMESTEPS=2000
OUT=artifacts/

python -m src.agents.train_agent --config $CONFIG --algo $ALGO --timesteps $TIMESTEPS --out_dir $OUT