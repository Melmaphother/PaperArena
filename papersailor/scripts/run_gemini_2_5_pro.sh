#!/bin/bash
# Run evaluation with Gemini-2.5 Pro

echo "🚀 Starting evaluation with Gemini-2.5 Pro"
python3 run_react.py \
  config=configs/base_config.yaml \
  model_name=gemini-2.5-pro