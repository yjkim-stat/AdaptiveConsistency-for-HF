export HF_TOKEN=TODO
export CACHE_DIR=TODO
export ATTN_IMPLEMENTATION='flash_attention_2'

export PYTHONPATH="$(pwd):$PYTHONPATH"

# Usage: bash scripts/run_adaptive_consistency.sh <stop_criteria> <stop_criteria_thresh>[optional]
stop_criteria=$1
# if stop_criteria_thresh is none set to -1
stop_criteria_thresh=${2:-"-1"}

# stop_criteria is one of:
# beta
# dirichlet
# entropy
# random
# majority

echo $stop_criteria
CUDA_DEVICE_ID=4

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE_ID python scripts/run_eval.py \
    --model google/gemma-3-27b-it\
    --dataset amc23 --answer_type latex \
    --max_gens 10 --temperature 1 \
    --prompt_type text --prompt_file step_reasoning
