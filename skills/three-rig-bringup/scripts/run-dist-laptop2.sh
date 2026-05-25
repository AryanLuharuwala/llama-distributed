#!/bin/bash
export DIST_DPP_PYTHONPATH=/home/boom/Desktop/Startup/DistibutedInference/llama-distributed/python
export DIST_PYTHON=/home/boom/.venv/bin/python3
export XDG_STATE_HOME=/tmp/dist-node-2/state
export CUDA_VISIBLE_DEVICES=""
cd /home/boom/Desktop/Startup/DistibutedInference/llama-distributed
exec ./build/dist-node --id 'shakti-fallback:laptop2' --pair 'distpool://pair?token=2d4a7a176f76be0165b43bc4f862ceb7f6e4a97e&server=ws://127.0.0.1:8080/ws/agent'
