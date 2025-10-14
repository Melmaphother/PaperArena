#!/bin/bash
# Run evaluation with Claude Sonnet 4

echo "🚀 Starting evaluation with Claude Sonnet 4"
python3 run_react.py \
  config=configs/base_config.yaml \
  model_name=claude-sonnet-4-20250514