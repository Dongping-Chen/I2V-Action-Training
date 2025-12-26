#!/bin/bash

# Wan2.2-TI2V-5B Action Controller Training Script

export DIFFSYNTH_ATTENTION_IMPLEMENTATION=torch

# DATASET_BASE=/fs/cml-projects/worldmodel/Self-Forcing/dataset_download/Sekai-Project/sekai-game-drone_v2_processed
# METADATA_PATH=${DATASET_BASE}/clips_metadata_diffsynth.csv
DATASET_BASE=""
METADATA_PATH=/fs/cml-projects/worldmodel/Self-Forcing/DiffSynth-Studio/ti2v_merge_sample10.csv
MODEL_DIR=/fs/cml-projects/worldmodel/hf_models/Wan2.2-TI2V-5B

# Periodic eval (batch inference during training)
EVAL_STEPS=${EVAL_STEPS:-1000}
EVAL_CSV=${EVAL_CSV:-${METADATA_PATH}}
EVAL_NUM_INFERENCE_STEPS=${EVAL_NUM_INFERENCE_STEPS:-50}
EVAL_CFG_SCALE=${EVAL_CFG_SCALE:-5.0}
EVAL_SEED=${EVAL_SEED:-42}
EVAL_FPS=${EVAL_FPS:-15}

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
    --action_injection_type "layer_add" \
    --action_initial_scale 0.1 \
    --learning_rate 1e-4 \
    --lr_warmup_ratio 0.05 \
    --lr_min_lr 1e-4 \
    --weight_decay 0.01 \
    --max_steps 10000 \
    --log_steps 10 \
    --save_steps 2000 \
    --save_full_state \
    --max_checkpoints 5 \
    --eval_steps "${EVAL_STEPS}" \
    --eval_csv "${EVAL_CSV}" \
    --eval_num_inference_steps "${EVAL_NUM_INFERENCE_STEPS}" \
    --eval_cfg_scale "${EVAL_CFG_SCALE}" \
    --eval_seed "${EVAL_SEED}" \
    --eval_fps "${EVAL_FPS}" \
    --use_wandb \
    --wandb_project "wan-ti2v-action" \
    --wandb_run_name "action-controller" \
    --gradient_accumulation_steps 4 \
    --height 480 \
    --width 832 \
    --num_frames 81 \
    --output_path "./models/train/Wan2.2-TI2V-5B_action_controller_layer_add_0.1_no_lora_10000" \
    "${RESUME_ARGS[@]}"
