#!/bin/bash
# Run evaluation with Kimi-K2

echo "🚀 Starting evaluation with Kimi-K2"
python3 run_react.py \
  config=configs/base_config.yaml \
  model_name=kimi-k2-250711
