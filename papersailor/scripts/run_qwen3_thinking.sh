#!/bin/bash
# Run evaluation with Qwen3-235B-A22B-Thinking

echo "🚀 Starting evaluation with Qwen3-235B-A22B-Thinking"
python3 run_react.py \
  config=configs/base_config.yaml \
  model_name=qwen/qwen3-235b-a22b-thinking-2507