#!/bin/bash
# Start a vLLM OpenAI-compatible server on GreatLakes

# Usage:
# 1. Edit MODEL_NAME
# 2. Fill in a Slurm class account you have access to
# 3. Submit an sbatch job
# 4. Watch the log until you see: `vLLM server available at: http://<NODE_IP>:8000/v1`
# 5. Pass the URL to `build_local_model()`

#SBATCH --job-name=vllm-server
#SBATCH --account=<Slurm account>
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00

MODEL_NAME="Qwen/Qwen3.5-4B"

module purge
module load cuda/12.1.1
module load python/3.11.5

mkdir -p logs

# Startup
NODE_ADDR=$(hostname -i)
echo "============================================="
echo "Starting vLLM server for model: ${MODEL_NAME}"
echo "Compute node: $(hostname) (${NODE_ADDR})"
echo "vLLM server available at: http://${NODE_ADDR}:8000/v1"
echo "============================================="

# Confirm vLLM exists
python -c "import vllm" 2>/dev/null || { echo "ERROR: vllm is not installed. Run: pip install vllm"; exit 1; }

# Start vLLM (in the background)
python -m vllm.entrypoints.openai.api_server --model "${MODEL_NAME}" --host 0.0.0.0 --port 8000 &

VLLM_PID=$!
echo "vLLM process PID: ${VLLM_PID}"

# Wait for the server
echo "Waiting for vLLM server to become ready..."
MAX_WAIT=300 # seconds
ELAPSED=0
INTERVAL=5

until curl -s "http://localhost:8000/health" > /dev/null 2>&1; do
    if [ ${ELAPSED} -ge ${MAX_WAIT} ]; then
        echo "ERROR: vLLM server failed to start within the timeout period. Exiting."
        exit 1
    fi
    sleep ${INTERVAL}
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "ERROR: vLLM process (PID ${VLLM_PID}) died unexpectedly. Check the logs above."
        exit 1
    fi
    ELAPSED=$((ELAPSED + INTERVAL))
    echo "Still waiting... (${ELAPSED}s elapsed)"
done

echo "vLLM server is ready!"
echo "Connect with: build_local_model('${MODEL_NAME}', 'http://${NODE_ADDR}:8000/v1')"

# Keep the server running until time limit
wait "${VLLM_PID}"
