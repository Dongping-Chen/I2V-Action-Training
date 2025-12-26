#!/usr/bin/env bash
# Wan2.2-TI2V-5B LoRA Training Script
# Dataset: Sekai drone video dataset

export DIFFSYNTH_ATTENTION_IMPLEMENTATION=torch

MODEL_DIR=/fs/cml-projects/worldmodel/hf_models/Wan2.1-T2V-1.3B
# /fs/cml-projects/worldmodel/hf_models/Wan2.2-TI2V-5B

# Periodic eval (batch inference during training)
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EVAL_STEPS=${EVAL_STEPS:-1000}
EVAL_CSV=${EVAL_CSV:-"${SCRIPT_DIR}/ti2v_merge_sample10.csv"}
EVAL_NUM_INFERENCE_STEPS=${EVAL_NUM_INFERENCE_STEPS:-50}
EVAL_CFG_SCALE=${EVAL_CFG_SCALE:-5.0}
EVAL_SEED=${EVAL_SEED:-42}
EVAL_FPS=${EVAL_FPS:-15}

# Optional resume from a checkpoint directory:
#   RESUME_FROM=./models/train/Wan2.2-TI2V-5B_merged_lora/checkpoints/step-00005000
RESUME_FROM="/fs/cml-projects/worldmodel/Self-Forcing/DiffSynth-Studio/models/train/Wan2.1-1.3B_lora_debug/checkpoints/step-00002000"
RESUME_ARGS=()
if [ -n "${RESUME_FROM}" ]; then
  RESUME_ARGS+=(--resume_from "${RESUME_FROM}")
fi

accelerate launch examples/wanvideo/model_training/train.py \
  --dataset_base_path "/fs/cml-projects/worldmodel/Self-Forcing/dataset_download/Sekai-Project/sekai-game-drone_v2_processed" \
  --dataset_metadata_path /fs/cml-projects/worldmodel/Self-Forcing/dataset_download/Sekai-Project/sekai-game-drone_v2_processed/clips_metadata_diffsynth.csv \
  --height 480 \
  --width 832 \
  --num_frames 81 \
  --dataset_num_workers 16 \
  --model_dir "${MODEL_DIR}" \
  --learning_rate 1e-4 \
  --lr_warmup_ratio 0.05 \
  --lr_min_lr 1e-4 \
  --weight_decay 0.01 \
  --max_steps 5000 \
  --log_steps 10 \
  --save_steps 1000 \
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
  --wandb_project "wan1.3b-i2v-lora" \
  --wandb_run_name "merged-lora-r64" \
  --output_path "./models/train/Wan2.1-1.3B_lora_debug" \
  --remove_prefix_in_ckpt "pipe.dit." \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 64 \
  --extra_inputs "input_image" \
  --force_i2v_fuse \
  "${RESUME_ARGS[@]}"
