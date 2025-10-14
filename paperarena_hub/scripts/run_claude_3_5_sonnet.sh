#!/bin/bash
# Run evaluation with Claude-3.5 Sonnet

echo "🚀 Starting evaluation with Claude-3.5 Sonnet"
python3 run_react.py \
  config=configs/base_config.yaml \
  model_name=claude-3-5-sonnet-20241022