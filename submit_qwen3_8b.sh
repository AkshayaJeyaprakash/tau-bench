#!/bin/bash
#SBATCH --job-name=tau_qwen3_8b_multiagent
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --gres=gpu:a100:3
#SBATCH --partition=public
#SBATCH --qos=class
#SBATCH --output=logs/tau_qwen3_8b_multiagent_%j.out
#SBATCH --error=logs/tau_qwen3_8b_multiagent_%j.err

set -e

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "Model: Qwen3-8B Multi-Agent"
echo "=========================================="

nvidia-smi

# ─── Paths ───────────────────────────────────────────────
SCRATCH=/scratch/npiduru1
PROJECT=$SCRATCH/tau-bench-project
HF_CACHE=$SCRATCH/hf_cache
VLLM_ENV=$HOME/miniconda3/envs/vllm_env
TAU_ENV=$HOME/miniconda3/envs/tau_bench

AGENT_MODEL="Qwen/Qwen3-8B"
USER_MODEL="meta-llama/Llama-3.1-8B-Instruct"

AGENT_PORT_1=8000
AGENT_PORT_2=8001
USER_PORT=8100

mkdir -p $PROJECT/logs
mkdir -p $HF_CACHE

export HF_HOME=$HF_CACHE
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
export HUGGINGFACE_TOKEN=$HF_TOKEN
export HUGGINGFACE_HUB_CACHE=$HF_CACHE
export OPENAI_API_KEY="dummy"

# ─── Step 1: Download models ─────────────────────────────
echo ""
echo ">>> Step 1: Downloading models if not cached..."
source $HOME/miniconda3/bin/activate vllm_env

$HOME/miniconda3/envs/vllm_env/bin/python -c "
from huggingface_hub import snapshot_download
import os
cache = os.environ['HF_HOME']
print('Downloading $AGENT_MODEL ...')
snapshot_download(repo_id='$AGENT_MODEL', cache_dir=cache)
print('Downloading $USER_MODEL ...')
snapshot_download(repo_id='$USER_MODEL', cache_dir=cache)
print('All downloads complete.')
"

# ─── Step 2: Start vLLM agent instance 1 (GPU 0) ─────────
echo ""
echo ">>> Step 2: Starting agent vLLM instance 1 on port $AGENT_PORT_1 (GPU 0)..."
CUDA_VISIBLE_DEVICES=0 $HOME/miniconda3/envs/vllm_env/bin/python -m vllm.entrypoints.openai.api_server --model $AGENT_MODEL --host 127.0.0.1 --port $AGENT_PORT_1 --download-dir $HF_CACHE --dtype bfloat16 --gpu-memory-utilization 0.90 --max-model-len 8192 --served-model-name Qwen3-8B > $PROJECT/logs/vllm_agent1_${SLURM_JOB_ID}.log 2>&1 &
AGENT_PID_1=$!

# ─── Step 3: Start vLLM agent instance 2 (GPU 1) ─────────
echo ">>> Step 3: Starting agent vLLM instance 2 on port $AGENT_PORT_2 (GPU 1)..."
CUDA_VISIBLE_DEVICES=1 $HOME/miniconda3/envs/vllm_env/bin/python -m vllm.entrypoints.openai.api_server --model $AGENT_MODEL --host 127.0.0.1 --port $AGENT_PORT_2 --download-dir $HF_CACHE --dtype bfloat16 --gpu-memory-utilization 0.90 --max-model-len 8192 --served-model-name Qwen3-8B > $PROJECT/logs/vllm_agent2_${SLURM_JOB_ID}.log 2>&1 &
AGENT_PID_2=$!

# ─── Step 4: Start vLLM user model (GPU 2) ───────────────
echo ">>> Step 4: Starting user vLLM instance on port $USER_PORT (GPU 2)..."
CUDA_VISIBLE_DEVICES=2 $HOME/miniconda3/envs/vllm_env/bin/python -m vllm.entrypoints.openai.api_server --model $USER_MODEL --host 127.0.0.1 --port $USER_PORT --download-dir $HF_CACHE --dtype bfloat16 --gpu-memory-utilization 0.90 --max-model-len 8192 --served-model-name Llama-3.1-8B-Instruct > $PROJECT/logs/vllm_user_${SLURM_JOB_ID}.log 2>&1 &
USER_PID=$!

# ─── Step 5: Wait for all 3 servers ──────────────────────
echo ""
echo ">>> Step 5: Waiting for all 3 vLLM servers to be ready..."

wait_for_server() {
    local port=$1
    local name=$2
    local max_wait=600
    local waited=0
    while [ $waited -lt $max_wait ]; do
        if curl -s http://127.0.0.1:$port/v1/models > /dev/null 2>&1; then
            echo "✓ $name on port $port is ready!"
            return 0
        fi
        sleep 10
        waited=$((waited + 10))
        echo "  Waiting for $name... (${waited}s)"
    done
    echo "✗ $name on port $port failed to start after ${max_wait}s"
    exit 1
}

wait_for_server $AGENT_PORT_1 "Agent instance 1"
wait_for_server $AGENT_PORT_2 "Agent instance 2"
wait_for_server $USER_PORT    "User model"

echo ""
echo ">>> All 3 vLLM servers are ready!"

# ─── Step 6: Run experiments ─────────────────────────────
echo ""
echo ">>> Step 6: Running experiments..."
source $HOME/miniconda3/bin/activate tau_bench
cd $PROJECT/tau-bench

export OPENAI_API_BASE="http://127.0.0.1:$AGENT_PORT_1/v1"
export USER_API_BASE="http://127.0.0.1:$USER_PORT/v1"

$HOME/miniconda3/envs/tau_bench/bin/python run.py \
    --agent-strategy multi-agent \
    --env retail \
    --model openai/Qwen3-8B \
    --model-provider openai \
    --user-model openai/Llama-3.1-8B-Instruct \
    --user-model-provider openai \
    --user-strategy llm \
    --max-concurrency 1 \
    --num-trials 5 \
    --task-ids 0 \
    --log-dir results

sleep 10

$HOME/miniconda3/envs/tau_bench/bin/python run.py \
    --agent-strategy multi-agent \
    --env airline \
    --model openai/Qwen3-8B \
    --model-provider openai \
    --user-model openai/Llama-3.1-8B-Instruct \
    --user-model-provider openai \
    --user-strategy llm \
    --max-concurrency 1 \
    --num-trials 5 \
    --task-ids 0 \
    --log-dir results

# ─── Cleanup ─────────────────────────────────────────────
echo ""
echo ">>> Cleaning up vLLM servers..."
kill $AGENT_PID_1 $AGENT_PID_2 $USER_PID 2>/dev/null || true

echo ""
echo "=========================================="
echo "Job complete: $(date)"
echo "Results in: $PROJECT/tau-bench/results/"
echo "=========================================="
