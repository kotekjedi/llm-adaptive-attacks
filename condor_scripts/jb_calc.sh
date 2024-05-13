#!/bin/bash

# Load environment
source /home/apanfilov/python/python/etc/profile.d/conda.sh
conda activate jailbreak_filter

# Define the working directory
export WORK_DIR=/home/apanfilov/jailbreak_filter/llm-adaptive-attacks
export CUDA_HOME=/is/software/nvidia/cuda-12.1
export LD_LIBRARY_PATH=/is/software/nvidia/cuda-12.1/lib64

cd $WORK_DIR || exit 1

# Receive parameters from command line
index=$1


# Construct the command
# cmd="python -u generate_test_cases.py \
#     --method_name $method_name \
#     --experiment_name $experiment_name \
#     --behaviors_path $behaviors_path \
#     --save_dir $save_dir --verbose \
#     --behavior_start_idx $start_idx \
#     --behavior_end_idx $end_idx"

cmd="python -u main.py \
    --n-iterations 10000 \
    --prompt-template 'refined_best' \
    --target-model 'gemma-7b' \
    --index $index \
    --n-tokens-adv 25 \
    --n-tokens-change-max 4 \
    --schedule_prob \
    --judge-max-n-calls 10 >> logs/exps_gemma-7b.log"


# Execute the command
eval $cmd
