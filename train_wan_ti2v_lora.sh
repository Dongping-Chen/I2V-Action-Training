#!/usr/bin/env bash
#SBATCH --job-name=wan-5B
#SBATCH --account=csd-sarahwie
#SBATCH --qos=csd-huge-long
#SBATCH --partition=csd-h200
#SBATCH --nodelist=vulcan46
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=480G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm_%j.out

set -euo pipefail

CONDA_ROOT="/fs/nexus-scratch/dongping/miniconda3"
if [[ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
  source "${CONDA_ROOT}/etc/profile.d/conda.sh"
else
  echo "[FATAL] conda.sh not found at ${CONDA_ROOT}" >&2
  exit 1
fi

module load gcc/11.2.0
conda activate diffsynth


export WANDB_API_KEY='ce2d58d19831ab0824ab0b36c3a78d4111f7f672'

# Wan2.2-TI2V-5B LoRA Training Script
# Dataset: Sekai drone video dataset

export DIFFSYNTH_ATTENTION_IMPLEMENTATION=torch

MODEL_DIR=/fs/cml-projects/worldmodel/hf_models/Wan2.2-TI2V-5B

# Periodic eval (batch inference during training)
EVAL_STEPS=${EVAL_STEPS:-1000}
EVAL_CSV=${EVAL_CSV:-/fs/cml-projects/worldmodel/Self-Forcing/DiffSynth-Studio/ti2v_merge_sample10.csv}
EVAL_NUM_INFERENCE_STEPS=${EVAL_NUM_INFERENCE_STEPS:-50}
EVAL_CFG_SCALE=${EVAL_CFG_SCALE:-5.0}
EVAL_SEED=${EVAL_SEED:-42}
EVAL_FPS=${EVAL_FPS:-15}

# Optional resume from a checkpoint directory:
#   RESUME_FROM=./models/train/Wan2.2-TI2V-5B_merged_lora/checkpoints/step-00005000
RESUME_FROM=${RESUME_FROM:-}
RESUME_ARGS=()
if [ -n "${RESUME_FROM}" ]; then
  RESUME_ARGS+=(--resume_from "${RESUME_FROM}")
fi

accelerate launch examples/wanvideo/model_training/train.py \
  --dataset_base_path "" \
  --dataset_metadata_path /fs/cml-projects/worldmodel/Self-Forcing/dataset_download/Sekai-Project/ti2v_merge.csv \
  --height 480 \
  --width 832 \
  --num_frames 81 \
  --dataset_num_workers 16 \
  --model_dir "${MODEL_DIR}" \
  --learning_rate 1e-4 \
  --lr_warmup_ratio 0.05 \
  --lr_min_lr 1e-4 \
  --weight_decay 0.01 \
  --max_steps 40000 \
  --log_steps 10 \
  --save_steps 5000 \
  --save_full_state \
  --max_checkpoints 3 \
  --eval_steps "${EVAL_STEPS}" \
  --eval_csv "${EVAL_CSV}" \
  --eval_num_inference_steps "${EVAL_NUM_INFERENCE_STEPS}" \
  --eval_cfg_scale "${EVAL_CFG_SCALE}" \
  --eval_seed "${EVAL_SEED}" \
  --eval_fps "${EVAL_FPS}" \
  --gradient_accumulation_steps 4 \
  --use_wandb \
  --wandb_project "wan-ti2v-lora" \
  --wandb_run_name "merged-lora-r32" \
  --output_path "./models/train/Wan2.2-TI2V-5B_merged_lora_40000" \
  --remove_prefix_in_ckpt "pipe.dit." \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --extra_inputs "input_image" \
  "${RESUME_ARGS[@]}"
