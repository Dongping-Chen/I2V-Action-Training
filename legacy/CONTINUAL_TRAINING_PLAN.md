# Continual Training, Checkpoints, and Local Model Directory Plan

## Goals
- Provide a complete save/load system (model + optimizer + scheduler + step) for resume training.
- Cover both I2V training (`examples/wanvideo/model_training/train.py`) and action training (`train_action.py`).
- Support merging LoRA into a base model and reusing it as the next training base.
- Replace `model_id_with_origin_paths` with a clean local directory input (no remote IDs).
- Add checkpoint retention and standardized layout for easier management.

## Non-goals
- Rewriting model architectures or pipeline logic.
- Changing dataset formats or sampling logic.
- Breaking existing CLI usage (new options should be additive).

## Current State (Implemented)
- `--model_dir` loads `model_index.json` and resolves local component paths (supports glob patterns).
- Checkpoints can include accelerator state for optimizer/scheduler/RNG when `--save_full_state` is enabled.
- `trainer_state.json` stores `global_step` and `epoch`; full args are saved to `train_config.json`.
- Resume restores full state and skips already-seen batches in the start epoch.
- Action training is disallowed to use LoRA; I2V training keeps LoRA support.
- `--seed` sets a global seed via `accelerate.utils.set_seed`.

## Proposed Design

### 1) Local Model Directory Standard
Define a local directory structure plus a manifest file for model components.

**Example layout:**
```
/base_models/Wan2.2-TI2V-5B-local/
  model_index.json
  dit/diffusion_pytorch_model-*.safetensors
  text_encoder/models_t5_umt5-xxl-enc-bf16.pth
  vae/Wan2.2_VAE.pth
  metadata.json
```

**model_index.json (example):**
```json
{
  "model_name": "wan2.2-ti2v-5b",
  "components": {
    "wan_video_dit": ["dit/diffusion_pytorch_model-00001-of-00002.safetensors", "dit/diffusion_pytorch_model-00002-of-00002.safetensors"],
    "wan_video_text_encoder": "text_encoder/models_t5_umt5-xxl-enc-bf16.pth",
    "wan_video_vae": "vae/Wan2.2_VAE.pth"
  }
}
```

**Loader behavior:**
- Add `--model_dir` (or `--base_model_dir`) to training scripts.
- If `--model_dir` is provided, read `model_index.json` and build a list of `ModelConfig(path=...)` entries.
- Keep `model_id_with_origin_paths` for backward compatibility.
- Component paths support glob patterns and are expanded at load time.

### 2) Checkpoint Format and Resume
Introduce a unified checkpoint directory for resume:
```
output_path/
  checkpoints/
    step-000010000/
      trainer_state.json
      train_config.json
      trainable.safetensors
      trainable_full.safetensors
      optimizer.pt
      scheduler.pt
      action_controller.safetensors (if action training)
```

**Key points:**
- Save lightweight training state: trainable weights + optimizer/scheduler (no full model weights, no RNG snapshot).
- Save trainable weights separately (current behavior) for easy inference.
- Record `global_step` and `epoch` in `trainer_state.json`.
- Save full args snapshot to `train_config.json`.

**Resume flow:**
- Add `--resume_from` to training scripts.
- If set, load `trainable_full.safetensors` into the training module (strict=False), then load `optimizer.pt` and `scheduler.pt`.
- Resume `global_step` and scheduler/optimizer state; continue from the next step (LR does not reset).
- Skip already-seen batches in the first resumed epoch based on `global_step`.

### 3) Checkpoint Retention
Implement `max_checkpoints` pruning:
- After each save, delete oldest checkpoint directories beyond `--max_checkpoints`.
- Use step numbers for deterministic ordering.

### 4) LoRA Merge for Next Stage Training
Provide a utility to fuse LoRA into a base model and export a new local model dir:
- Load base model from `--model_dir`.
- Load LoRA via `--lora_path`.
- Use `GeneralLoRALoader.fuse_lora_to_base_model()` on the target module (e.g., `pipe.dit`).
- Save a new base model directory with `model_index.json`.
- This new directory can then be used via `--model_dir` for the next training stage.

### 5) Training Script Integration
Update both I2V and action training scripts:
- Add `--model_dir` and `--resume_from`.
- Add `--save_full_state` (toggle for accelerator state save) and `--max_checkpoints` support.
- Add `--seed` for deterministic runs and reproducible resume.
- Keep action-only options intact (e.g., `--action_initial_scale`).
- Disallow LoRA args in action training (explicit error).

## Implementation Steps
1) Add a helper to read `model_index.json` and build `ModelConfig` list. (done)
2) Extend `parse_model_configs` to accept `--model_dir` (and ignore `model_id_with_origin_paths` if set). (done)
3) Add checkpoint manager (new or in `logger.py`) with retention logic. (done)
4) Update `runner.py` to save/restore accelerator state and `trainer_state.json`. (done)
5) Update `train.py` and `train_action.py` CLI args and wiring for `--model_dir` and `--resume_from`. (done)
6) Add a “merge LoRA to base model” utility script or CLI mode. (done)
7) Document the new local model directory format and resume workflow. (done)

## Migration Plan
- Existing scripts keep working with `model_id_with_origin_paths`.
- New local directory + manifest is optional and can be phased in gradually.
- Action training continues to save `action_controller.safetensors` for inference.
- Action training rejects LoRA args to avoid mixing tasks.

## Validation Checklist
- Resume from checkpoint continues training with the same loss curve (no LR reset).
- Checkpoint pruning keeps only the most recent N directories.
- `--model_dir` loads the correct components without remote downloads.
- LoRA merge produces a base model directory that can be re-used for training.
- Resume skips previously seen batches in the first epoch.
- `train_config.json` is saved alongside checkpoints.
