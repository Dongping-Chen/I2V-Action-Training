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

# Wan2.2-TI2V-5B Action Controller Training Script

export DIFFSYNTH_ATTENTION_IMPLEMENTATION=flash_attention_2
ACTION_CROSS_ATTN_TRAIN_MODE=${ACTION_CROSS_ATTN_TRAIN_MODE:-lora}

# DATASET_BASE=/fs/cml-projects/worldmodel/Self-Forcing/dataset_download/Sekai-Project/sekai-game-drone_v2_processed
# METADATA_PATH=${DATASET_BASE}/clips_metadata_diffsynth.csv
DATASET_BASE=""
METADATA_PATH=/fs/cml-projects/worldmodel/Self-Forcing/dataset_download/Sekai-Project/ti2v_merge.csv
MODEL_DIR=/fs/cml-projects/worldmodel/Self-Forcing/DiffSynth-Studio/models/Wan2.2-TI2V-5B-lora
EVAL_CSV=/fs/cml-projects/worldmodel/Self-Forcing/DiffSynth-Studio/ti2v_merge_sample10_new.csv

# Optional resume from a checkpoint directory:
#   RESUME_FROM=./models/train/Wan2.2-TI2V-5B_action_controller/checkpoints/step-00001000
RESUME_FROM=${RESUME_FROM:-}
RESUME_ARGS=()
if [ -n "${RESUME_FROM}" ]; then
    RESUME_ARGS+=(--resume_from "${RESUME_FROM}")
fi

accelerate launch examples/wanvideo/model_training/train_action.py \
    --dataset_base_path "${DATASET_BASE}" \
    --dataset_metadata_path "${METADATA_PATH}" \
    --data_file_keys "video" \
    --dataset_num_workers 16 \
    --model_dir "${MODEL_DIR}" \
    --trainable_models "action_controller" \
    --action_controller_config "wan_ti2v_5b" \
    --action_injection_type "cross_attn" \
    --action_cross_attn_train_mode "${ACTION_CROSS_ATTN_TRAIN_MODE}" \
    --action_initial_scale 1 \
    --lora_base_model "action_controller" \
    --lora_target_modules "q,k,v,o" \
    --lora_rank 64 \
    --learning_rate 1e-4 \
    --lr_warmup_ratio 0.05 \
    --lr_min_lr 1e-4 \
    --weight_decay 0.01 \
    --max_steps 40000 \
    --log_steps 10 \
    --save_steps 5000 \
    --save_full_state \
    --max_checkpoints 5 \
    --eval_steps 1000 \
    --eval_csv "${EVAL_CSV}" \
    --eval_num_inference_steps 50 \
    --eval_cfg_scale 5 \
    --eval_seed 42 \
    --eval_fps 16 \
    --use_wandb \
    --wandb_project "wan-ti2v-action" \
    --wandb_run_name "action-controller" \
    --gradient_accumulation_steps 4 \
    --height 480 \
    --width 832 \
    --num_frames 81 \
    --output_path "./models/train/Wan2.2-TI2V-5B_action_controller_cross_attn_lora_r64_gate1_40000" \
    "${RESUME_ARGS[@]}"
