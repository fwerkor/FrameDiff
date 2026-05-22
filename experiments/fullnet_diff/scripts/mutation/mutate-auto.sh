#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
WORKSPACE_ROOT=$(cd -- "${REPO_ROOT}/.." && pwd)

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"

NPUS_PER_NODE="${NPUS_PER_NODE:-1}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-6000}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
SEQ_LEN="${SEQ_LEN:-1024}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${REPO_ROOT}/assets/runtime/tokenizers/baichuan2}"
MODEL_CONFIG_DIR="${MODEL_CONFIG_DIR:-${WORKSPACE_ROOT}/frame_diff_common/model_configs}"
MUTATION_SCHEMA_PATH="${MUTATION_SCHEMA_PATH:-${REPO_ROOT}/assets/runtime/configs/mutation_schema.yaml}"
MUTATE_ENTRY="${MUTATE_ENTRY:-${REPO_ROOT}/utils/runtime/mutate_and_forward/mutate_graph-auto.py}"

MASTER_PORT="$(
python - "${MASTER_PORT}" <<'PY'
import socket
import sys

port = int(sys.argv[1])

def is_free(value: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("localhost", value))
        except OSError:
            return False
    return True

if not is_free(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("localhost", 0))
        port = sock.getsockname()[1]

print(port)
PY
)"

DEFAULT_MUTATE_ARGS="-c ${MODEL_CONFIG_DIR} -r 100 --mutnm 2 -n 1 -m ${MODEL_CONFIG_DIR}/qwen2.yaml --args_path ${MUTATION_SCHEMA_PATH}"
MUTATE_ARGS="${MUTATE_ARGS:-${DEFAULT_MUTATE_ARGS}}"

distributed_args=(
    --nproc_per_node "${NPUS_PER_NODE}"
    --nnodes "${NNODES}"
    --node_rank "${NODE_RANK}"
    --master_addr "${MASTER_ADDR}"
    --master_port "${MASTER_PORT}"
)

gpt_args=(
    --num-layers 16
    --hidden-size 928
    --ffn-hidden-size 1712
    --num-attention-heads 8
    --tokenizer-type PretrainedFromHF
    --tokenizer-name-or-path "${TOKENIZER_PATH}"
    --seq-length "${SEQ_LEN}"
    --max-position-embeddings "${SEQ_LEN}"
    --micro-batch-size 1
    --global-batch-size 8
)

echo "Using MASTER_PORT=${MASTER_PORT}"

# shellcheck disable=SC2086
torchrun "${distributed_args[@]}" "${MUTATE_ENTRY}" "${gpt_args[@]}" ${MUTATE_ARGS}
