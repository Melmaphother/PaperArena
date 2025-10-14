#!/bin/bash
# Run evaluation with Qwen3-235B-A22B-Instruct

echo "🚀 Starting evaluation with Qwen3-235B-A22B-Instruct"
python3 run_react.py \
  config=configs/base_config.yaml \
  model_name=Qwen/Qwen3-235B-A22B-Instruct-2507