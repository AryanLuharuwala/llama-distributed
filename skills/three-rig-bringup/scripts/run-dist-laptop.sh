#!/bin/bash
export DIST_DPP_PYTHONPATH=/home/boom/Desktop/Startup/DistibutedInference/llama-distributed/python
export DIST_PYTHON=/home/boom/.venv/bin/python3
cd /home/boom/Desktop/Startup/DistibutedInference/llama-distributed
exec ./build/dist-node --pair 'distpool://pair?token=4371106df17ea5ab10a1961b2bd91bf4716340f8&server=ws://127.0.0.1:8080/ws/agent'
