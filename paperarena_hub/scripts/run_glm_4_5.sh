#!/bin/bash
# Run evaluation with GLM-4.5

echo "🚀 Starting evaluation with GLM-4.5"
python3 run_react.py \
  config=configs/base_config.yaml \
  model_name=glm-4.5