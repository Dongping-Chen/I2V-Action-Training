import itertools
import json
import os
import math
import torch
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed
from .training_module import DiffusionTrainingModule
from .logger import ModelLogger
from ..core import load_state_dict

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def _align_action_length(action_magnitude: torch.Tensor, target_len):
    if target_len is None:
        return action_magnitude
    current_len = int(action_magnitude.shape[1])
    target_len = int(target_len)
    if current_len == target_len:
        return action_magnitude
    raise ValueError(f"Action length mismatch: expected {target_len}, got {current_len}.")


def _parse_action_value(action_value) -> torch.Tensor | None:
    if action_value is None:
        return None
    if isinstance(action_value, (list, tuple)):
        action = action_value
    else:
        raw = str(action_value).strip()
        if not raw:
            return None
        try:
            action = json.loads(raw)
        except Exception:
            import ast
            action = ast.literal_eval(raw)
    action = torch.tensor(action, dtype=torch.float32)
    if action.dim() == 4 and action.shape[2] == 2:
        action_magnitude = action[:, :, 1, :]
    elif action.dim() == 3 and action.shape[1] == 2 and action.shape[2] == 8:
        action_magnitude = action.unsqueeze(0)[:, :, 1, :]
    elif action.dim() == 2:
        action_magnitude = action.unsqueeze(0)
    else:
        action_magnitude = action
    if action_magnitude.dim() != 3 or action_magnitude.shape[-1] != 8:
        raise ValueError(f"Unexpected action shape: {tuple(action.shape)} (expected [T,8] or [B,T,8])")
    return action_magnitude


def _run_periodic_eval(
    accelerator: Accelerator,
    model: DiffusionTrainingModule,
    args,
    global_step: int,
    shard_by_rank: bool = False,
):
    if args is None:
        return

    eval_steps = getattr(args, "eval_steps", 0)
    if eval_steps <= 0 or global_step % eval_steps != 0:
        return

    eval_csv = getattr(args, "eval_csv", "") or ""
    if not eval_csv:
        return
    if not os.path.isabs(eval_csv):
        eval_csv = os.path.join(os.getcwd(), eval_csv)
    if not os.path.isfile(eval_csv):
        if accelerator.is_main_process:
            print(f"[eval] Warning: eval CSV not found, skipping: {eval_csv}")
        return
    if not shard_by_rank and not accelerator.is_main_process:
        return

    import csv
    from pathlib import Path
    from PIL import Image

    from ..core.data.operators import ImageCropAndResize
    from ..utils.data import VideoData, save_video

    unwrapped = accelerator.unwrap_model(model)
    pipe = getattr(unwrapped, "pipe", None)
    if pipe is None:
        if accelerator.is_main_process:
            print("[eval] Warning: model has no .pipe attribute, skipping eval.")
        return

    scheduler = getattr(pipe, "scheduler", None)
    scheduler_state = None
    if scheduler is not None:
        scheduler_state = {
            "sigmas": getattr(scheduler, "sigmas", None),
            "timesteps": getattr(scheduler, "timesteps", None),
            "training": getattr(scheduler, "training", False),
            "linear_timesteps_weights": getattr(scheduler, "linear_timesteps_weights", None),
        }

    pipe_child_training_states = {
        name: child.training
        for name, child in pipe.named_children()
        if child is not None
    }
    pipe.eval()

    use_action = getattr(pipe, "action_controller", None) is not None
    eval_root = os.path.join(
        getattr(args, "output_path", "."),
        getattr(args, "eval_output_dirname", "eval"),
        f"step-{global_step:08d}",
    )
    os.makedirs(eval_root, exist_ok=True)

    base_path = getattr(args, "eval_base_path", "") or ""
    base_path = Path(base_path) if base_path else Path(eval_csv).parent

    def resolve_path(raw_path):
        raw_path = (raw_path or "").strip()
        if not raw_path:
            return None
        p = Path(raw_path)
        if p.is_absolute():
            return p
        return base_path / p

    image_processor = ImageCropAndResize(
        height=getattr(args, "height", None),
        width=getattr(args, "width", None),
        max_pixels=getattr(args, "max_pixels", None),
        height_division_factor=getattr(pipe, "height_division_factor", 16),
        width_division_factor=getattr(pipe, "width_division_factor", 16),
    )

    eval_num_frames = getattr(args, "num_frames", 81)
    eval_num_inference_steps = getattr(args, "eval_num_inference_steps", 50)
    eval_cfg_scale = getattr(args, "eval_cfg_scale", 5.0)
    eval_seed = getattr(args, "eval_seed", 42)
    eval_fps = getattr(args, "eval_fps", 15)
    eval_negative_prompt = getattr(args, "eval_negative_prompt", "") or ""

    time_division_factor = getattr(pipe, "time_division_factor", 4)
    time_division_remainder = getattr(pipe, "time_division_remainder", 1)
    expected_action_len = None
    if time_division_factor and eval_num_frames >= time_division_remainder:
        expected_action_len = max(0, (eval_num_frames - time_division_remainder) // time_division_factor)

    action_scale = getattr(unwrapped, "action_magnitude_scale", None)
    if action_scale is None:
        action_scale = torch.tensor([100, 100, 100, 100, 1, 1, 1, 1], dtype=torch.float32)
    elif not torch.is_tensor(action_scale):
        action_scale = torch.tensor(action_scale, dtype=torch.float32)
    action_scale = action_scale.detach().to(dtype=torch.float32, device="cpu")

    rank = accelerator.process_index
    world_size = accelerator.num_processes
    manifest = []

    try:
        overlay_actions_fn = None
        action_names = None
        if use_action:
            import cv2
            import numpy as np

            action_names = [
                "Forward",
                "Back",
                "Left",
                "Right",
                "Yaw Left",
                "Yaw Right",
                "Pitch Up",
                "Pitch Down",
            ]

            def draw_translation_controls(frame, base_x, base_y, size, actions):
                gap = int(size * 1.5)
                btn_radius = size // 2 + 6
                arrow_half_len = size // 3
                buttons = [
                    ("Forward", base_x, base_y - gap, (0, -1)),
                    ("Left", base_x - gap, base_y, (-1, 0)),
                    ("Back", base_x, base_y, (0, 1)),
                    ("Right", base_x + gap, base_y, (1, 0)),
                ]
                overlay = frame.copy()
                for name, bx, by, _ in buttons:
                    cv2.circle(overlay, (bx, by), btn_radius, (30, 30, 30), -1)
                frame[:] = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
                for name, bx, by, direction in buttons:
                    active = name in actions
                    dx, dy = direction
                    start_pt = (bx - dx * arrow_half_len, by - dy * arrow_half_len)
                    end_pt = (bx + dx * arrow_half_len, by + dy * arrow_half_len)
                    color = (255, 255, 255) if active else (60, 60, 60)
                    thickness = 3 if active else 2
                    cv2.arrowedLine(frame, start_pt, end_pt, color, thickness, tipLength=0.4)

            def draw_rotation_indicator(frame, position, size, active, direction):
                x, y = position
                dx, dy = direction
                arrow_half_len = size // 3
                start_pt = (x - dx * arrow_half_len, y - dy * arrow_half_len)
                end_pt = (x + dx * arrow_half_len, y + dy * arrow_half_len)
                color = (255, 255, 255) if active else (40, 40, 40)
                thickness = 3 if active else 2
                cv2.arrowedLine(frame, start_pt, end_pt, color, thickness, tipLength=0.4)

            def overlay_actions_on_frame(frame, actions, sz_trans, sz_rot):
                h, w = frame.shape[:2]
                base_x = int(0.12 * w)
                base_y = int(0.85 * h)
                draw_translation_controls(frame, base_x, base_y, sz_trans, actions)
                edge_margin = int(0.06 * min(w, h))
                draw_rotation_indicator(frame, (w // 2, edge_margin + sz_rot // 2), sz_rot, "Pitch Up" in actions, (0, -1))
                draw_rotation_indicator(frame, (w // 2, h - edge_margin - sz_rot // 2), sz_rot, "Pitch Down" in actions, (0, 1))
                draw_rotation_indicator(frame, (edge_margin + sz_rot // 2, h // 2), sz_rot, "Yaw Left" in actions, (-1, 0))
                draw_rotation_indicator(frame, (w - edge_margin - sz_rot // 2, h // 2), sz_rot, "Yaw Right" in actions, (1, 0))

            def overlay_actions_on_frames(frames, frame_actions):
                if not frames:
                    return frames
                sample = np.array(frames[0]) if isinstance(frames[0], Image.Image) else frames[0]
                height, width = sample.shape[:2]
                sz_trans = max(36, int(min(width, height) * 0.07))
                sz_rot = max(44, int(min(width, height) * 0.08))
                output = []
                for idx, frame in enumerate(frames):
                    frame_np = np.array(frame) if isinstance(frame, Image.Image) else frame.copy()
                    actions = frame_actions[idx] if idx < len(frame_actions) else set()
                    overlay_actions_on_frame(frame_np, actions, sz_trans, sz_rot)
                    output.append(frame_np)
                return output

            overlay_actions_fn = overlay_actions_on_frames

        with open(eval_csv, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row_idx, row in enumerate(reader):
                if shard_by_rank and (row_idx % world_size) != rank:
                    continue

                prompt = row.get("prompt") or ""
                video_path = resolve_path(row.get("video"))
                input_image_path = resolve_path(row.get("input_image"))
                action_value = row.get("action")

                out_prefix = f"{row_idx:04d}"
                out_video_path = os.path.join(eval_root, f"{out_prefix}_{'action' if use_action else 'ti2v'}.mp4")
                out_input_path = os.path.join(eval_root, f"{out_prefix}_input.jpg")
                out_overlay_path = os.path.join(eval_root, f"{out_prefix}_action_overlay.mp4") if use_action else None

                record = {
                    "row_idx": row_idx,
                    "rank": rank,
                    "world_size": world_size,
                    "video": str(video_path) if video_path else None,
                    "input_image": str(input_image_path) if input_image_path else None,
                    "seed": int(eval_seed) + int(row_idx),
                    "output_video": out_video_path,
                }

                try:
                    if input_image_path and input_image_path.exists():
                        input_image = Image.open(input_image_path).convert("RGB")
                    elif video_path and video_path.exists():
                        input_image = VideoData(str(video_path))[0]
                    else:
                        raise FileNotFoundError(f"Missing input_image and video for row {row_idx}")

                    input_image = image_processor(input_image)
                    input_image.save(out_input_path)

                    height = int(input_image.size[1])
                    width = int(input_image.size[0])

                    gen_kwargs = dict(
                        prompt=prompt,
                        negative_prompt=eval_negative_prompt,
                        input_image=input_image,
                        height=height,
                        width=width,
                        num_frames=eval_num_frames,
                        num_inference_steps=eval_num_inference_steps,
                        cfg_scale=eval_cfg_scale,
                        seed=int(eval_seed) + int(row_idx),
                        rand_device=getattr(pipe, "device", "cpu"),
                        tiled=True,
                        progress_bar_cmd=(lambda x: x),
                    )

                    if use_action:
                        action_magnitude = _parse_action_value(action_value)
                        if action_magnitude is None:
                            raise ValueError(f"Missing action for row {row_idx}")
                        action_magnitude = _align_action_length(action_magnitude, expected_action_len)
                        action_magnitude_raw = action_magnitude.detach().clone()
                        action_magnitude = action_magnitude * action_scale.to(action_magnitude.device, action_magnitude.dtype)
                        gen_kwargs["action_magnitude"] = action_magnitude

                    with torch.no_grad():
                        video = pipe(**gen_kwargs)
                    save_video(video, out_video_path, fps=eval_fps, quality=5, progress=False)
                    if use_action and overlay_actions_fn is not None and out_overlay_path is not None:
                        threshold = 1e-6
                        action_sets = []
                        for step in action_magnitude_raw[0]:
                            active = {action_names[i] for i, value in enumerate(step.tolist()) if value > threshold}
                            action_sets.append(active)
                        frame_actions = []
                        for frame_id in range(len(video)):
                            if frame_id < time_division_remainder or time_division_factor <= 0:
                                frame_actions.append(set())
                                continue
                            step_id = (frame_id - time_division_remainder) // time_division_factor
                            if 0 <= step_id < len(action_sets):
                                frame_actions.append(action_sets[int(step_id)])
                            else:
                                frame_actions.append(set())
                        overlay_frames = overlay_actions_fn(video, frame_actions)
                        save_video(overlay_frames, out_overlay_path, fps=eval_fps, quality=5, progress=False)
                    record["ok"] = True
                except Exception as e:
                    record["ok"] = False
                    record["error"] = str(e)
                manifest.append(record)
    finally:
        for name, child in pipe.named_children():
            if child is None or name not in pipe_child_training_states:
                continue
            if pipe_child_training_states[name]:
                child.train()
            else:
                child.eval()
        if scheduler is not None and scheduler_state is not None:
            if scheduler_state["sigmas"] is not None:
                scheduler.sigmas = scheduler_state["sigmas"]
            if scheduler_state["timesteps"] is not None:
                scheduler.timesteps = scheduler_state["timesteps"]
            scheduler.training = scheduler_state["training"]
            if scheduler_state["linear_timesteps_weights"] is not None:
                scheduler.linear_timesteps_weights = scheduler_state["linear_timesteps_weights"]

    manifest_path = os.path.join(eval_root, f"manifest_rank{rank:02d}.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    if accelerator.device.type == "cuda":
        torch.cuda.empty_cache()

    if shard_by_rank:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            print(f"[eval] step={global_step} done, wrote {eval_root}")
    elif accelerator.is_main_process:
        print(f"[eval] step={global_step} done, wrote {eval_root}")


def _safe_scalar(tensor: torch.Tensor):
    if tensor.numel() != 1:
        raise ValueError(f"Expected scalar tensor, got shape={tuple(tensor.shape)}")
    return float(tensor.detach().float().cpu().item())


def _summarize_action_controller_gates(model: torch.nn.Module):
    out = {}
    for name, param in model.named_parameters():
        if not name.startswith("pipe.action_controller."):
            continue
        short = name[len("pipe.action_controller."):]
        if short == "action_patch_gate":
            out["param/action_controller/action_patch_gate"] = _safe_scalar(param.data)
            if param.grad is not None:
                out["grad/action_controller/action_patch_gate"] = _safe_scalar(param.grad)
        elif short == "action_gate":
            data = param.data.detach().float().cpu()
            out["param/action_controller/action_gate_mean"] = float(data.mean().item())
            out["param/action_controller/action_gate_min"] = float(data.min().item())
            out["param/action_controller/action_gate_max"] = float(data.max().item())
            if param.grad is not None:
                grad = param.grad.detach().float().cpu()
                out["grad/action_controller/action_gate_mean"] = float(grad.mean().item())
                out["grad/action_controller/action_gate_min"] = float(grad.min().item())
                out["grad/action_controller/action_gate_max"] = float(grad.max().item())
    return out


def _summarize_trainable_grad_norms(model: torch.nn.Module):
    out = {}
    total_sq = 0.0
    ac_sq = 0.0
    lora_sq = 0.0
    other_sq = 0.0
    trainable_tensors = 0
    tensors_with_grad = 0

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        trainable_tensors += 1
        if param.grad is None:
            continue
        tensors_with_grad += 1
        g_norm = float(param.grad.detach().float().norm().item())
        total_sq += g_norm * g_norm
        if name.startswith("pipe.action_controller."):
            ac_sq += g_norm * g_norm
        elif "lora_" in name:
            lora_sq += g_norm * g_norm
        else:
            other_sq += g_norm * g_norm

    out["grad/total_norm"] = math.sqrt(total_sq)
    out["grad/action_controller/total_norm"] = math.sqrt(ac_sq)
    out["grad/lora/total_norm"] = math.sqrt(lora_sq)
    out["grad/other/total_norm"] = math.sqrt(other_sq)
    out["grad/trainable_tensors"] = trainable_tensors
    out["grad/tensors_with_grad"] = tensors_with_grad
    out["grad/tensors_with_grad_frac"] = (
        float(tensors_with_grad) / float(trainable_tensors) if trainable_tensors else 0.0
    )
    return out


def _load_train_config(resume_from):
    config_path = os.path.join(resume_from, "train_config.json")
    if not os.path.isfile(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_checkpoint_weights(resume_from, trainer_state, train_config):
    candidates = []
    if trainer_state and trainer_state.get("weights_name"):
        candidates.append(trainer_state["weights_name"])
    trainable_models = (train_config or {}).get("trainable_models") or ""
    if "action_controller" in trainable_models.split(","):
        candidates.append("action_controller.safetensors")
    if (train_config or {}).get("lora_base_model"):
        candidates.append("lora.safetensors")
    candidates += [
        "trainable.safetensors",
        "trainable_full.safetensors",
    ]
    seen = set()
    for name in candidates:
        if not name or name in seen:
            continue
        seen.add(name)
        path = os.path.join(resume_from, name)
        if os.path.isfile(path):
            return path
    return None


def _infer_resume_prefix(trainer_state, train_config):
    prefix = None
    if trainer_state:
        prefix = trainer_state.get("remove_prefix_in_ckpt") or None
    if not prefix and train_config:
        trainable_models = train_config.get("trainable_models") or ""
        if "action_controller" in trainable_models.split(","):
            prefix = "pipe.action_controller."
        lora_base_model = train_config.get("lora_base_model") or ""
        if not prefix and lora_base_model:
            prefix = f"pipe.{lora_base_model}."
        if not prefix:
            prefix = train_config.get("remove_prefix_in_ckpt") or None
    if prefix and not prefix.endswith("."):
        prefix = prefix + "."
    return prefix


def _apply_resume_prefix(state_dict, prefix):
    if not prefix:
        return state_dict
    updated = {}
    for key, value in state_dict.items():
        if key.startswith("pipe."):
            updated[key] = value
        else:
            updated[prefix + key] = value
    return updated


def launch_training_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    max_steps: int = None,
    log_steps: int = 10,
    args = None,
):
    if args is not None and getattr(args, "seed", None) is not None:
        set_seed(args.seed)
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
        max_steps = getattr(args, 'max_steps', None)
        log_steps = getattr(args, 'log_steps', 10)

    # Initialize wandb
    use_wandb = getattr(args, 'use_wandb', False) if args else False
    if use_wandb and accelerator.is_main_process:
        if not WANDB_AVAILABLE:
            print("Warning: wandb not installed, disabling wandb logging")
            use_wandb = False
        else:
            wandb_project = getattr(args, 'wandb_project', 'diffsynth-training')
            wandb_run_name = getattr(args, 'wandb_run_name', None)
            wandb_config = {
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "max_steps": max_steps,
                "num_epochs": num_epochs,
                "lr_warmup_steps": getattr(args, 'lr_warmup_steps', None),
                "lr_warmup_ratio": getattr(args, 'lr_warmup_ratio', None),
                "lr_min_lr": getattr(args, 'lr_min_lr', None),
                "lora_rank": getattr(args, 'lora_rank', None),
                "lora_target_modules": getattr(args, 'lora_target_modules', None),
                "height": getattr(args, 'height', None),
                "width": getattr(args, 'width', None),
                "num_frames": getattr(args, 'num_frames', None),
            }
            wandb.init(project=wandb_project, name=wandb_run_name, config=wandb_config)

    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    total_steps = max_steps if max_steps is not None else num_epochs * len(dataloader)
    warmup_steps = getattr(args, 'lr_warmup_steps', None) if args else None
    if warmup_steps is None:
        warmup_ratio = getattr(args, 'lr_warmup_ratio', 0.03) if args else 0.03
        warmup_steps = int(total_steps * warmup_ratio)
    warmup_steps = max(0, min(warmup_steps, max(total_steps - 1, 0)))
    min_lr = getattr(args, 'lr_min_lr', 0.0) if args else 0.0
    min_lr = 0.0 if min_lr is None else min_lr
    min_ratio = (min_lr / learning_rate) if learning_rate > 0 else 0.0
    min_ratio = max(0.0, min(min_ratio, 1.0))

    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)

    def lr_lambda(step):
        if total_steps <= 0:
            return 1.0
        if warmup_steps > 0 and step < warmup_steps:
            warmup_scale = float(step + 1) / float(warmup_steps)
            # Warm up from 0 -> 1 regardless of `min_lr`.
            # `min_lr` is treated as the cosine decay floor after warmup.
            return warmup_scale
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        cosine_scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine_scale

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

    def _move_optimizer_state_to_device(opt, device):
        inner = getattr(opt, "optimizer", opt)
        if not hasattr(inner, "state"):
            return
        for state in inner.state.values():
            if not isinstance(state, dict):
                continue
            for k, v in list(state.items()):
                if torch.is_tensor(v):
                    state[k] = v.to(device)
                elif isinstance(v, dict):
                    for kk, vv in list(v.items()):
                        if torch.is_tensor(vv):
                            v[kk] = vv.to(device)

    global_step = 0
    start_epoch = 0
    resume_from = getattr(args, "resume_from", None) if args else None
    if resume_from:
        trainer_state_path = os.path.join(resume_from, "trainer_state.json")
        if not os.path.isfile(trainer_state_path):
            raise FileNotFoundError(f"trainer_state.json not found in {resume_from}")
        with open(trainer_state_path, "r", encoding="utf-8") as f:
            trainer_state = json.load(f)
        train_config = _load_train_config(resume_from)
        global_step = int(trainer_state.get("global_step", 0))
        start_epoch = int(trainer_state.get("epoch") or 0)
        model_logger.set_num_steps(global_step)
        model_logger.last_checkpoint_step = global_step

        accelerator_state_path = os.path.join(resume_from, "accelerator_state")
        if os.path.isdir(accelerator_state_path):
            accelerator.load_state(accelerator_state_path)
        else:
            trainable_path = _resolve_checkpoint_weights(resume_from, trainer_state, train_config)
            if trainable_path is None:
                raise FileNotFoundError(f"trainable weights not found in {resume_from}")
            trainable_state = load_state_dict(trainable_path, device="cpu")
            prefix = _infer_resume_prefix(trainer_state, train_config)
            trainable_state = _apply_resume_prefix(trainable_state, prefix)
            accelerator.unwrap_model(model).load_state_dict(trainable_state, strict=False)

            optimizer_path = os.path.join(resume_from, "optimizer.pt")
            if not os.path.isfile(optimizer_path):
                raise FileNotFoundError(f"optimizer.pt not found in {resume_from}")
            optimizer_state = torch.load(optimizer_path, map_location="cpu")
            optimizer.load_state_dict(optimizer_state)
            _move_optimizer_state_to_device(optimizer, accelerator.device)

            scheduler_path = os.path.join(resume_from, "scheduler.pt")
            if os.path.isfile(scheduler_path):
                scheduler_state = torch.load(scheduler_path, map_location="cpu")
                try:
                    scheduler.load_state_dict(scheduler_state)
                except Exception as e:
                    if accelerator.is_main_process:
                        print(f"Warning: failed to load scheduler.pt ({e}). Falling back to global_step.")
                    setattr(scheduler, "last_epoch", global_step - 1)
            else:
                setattr(scheduler, "last_epoch", global_step - 1)

    steps_per_epoch = len(dataloader) if hasattr(dataloader, "__len__") else None
    resume_step_in_epoch = 0
    if resume_from and steps_per_epoch:
        expected_epoch = global_step // steps_per_epoch
        if start_epoch != expected_epoch:
            if accelerator.is_main_process:
                print(
                    f"Warning: resume epoch mismatch (checkpoint epoch {start_epoch}, "
                    f"computed epoch {expected_epoch}). Using computed epoch."
                )
            start_epoch = expected_epoch
        resume_step_in_epoch = global_step - start_epoch * steps_per_epoch

    running_loss = 0.0
    should_stop = False

    # Use max_steps mode if specified
    if max_steps is not None:
        num_epochs = 999999  # Effectively infinite epochs, controlled by max_steps
        progress_bar = tqdm(total=max_steps, desc="Training")
        if global_step > 0:
            progress_bar.update(global_step)

    if max_steps is not None and global_step >= max_steps:
        should_stop = True

    last_epoch = start_epoch
    for epoch_id in range(start_epoch, num_epochs):
        if should_stop:
            break
        last_epoch = epoch_id
        epoch_dataloader = dataloader
        if resume_step_in_epoch and epoch_id == start_epoch:
            if hasattr(accelerator, "skip_first_batches"):
                epoch_dataloader = accelerator.skip_first_batches(dataloader, resume_step_in_epoch)
            else:
                epoch_dataloader = itertools.islice(dataloader, resume_step_in_epoch, None)
        if max_steps is not None:
            epoch_iterator = epoch_dataloader
        else:
            if resume_step_in_epoch and epoch_id == start_epoch and steps_per_epoch is not None:
                remaining = max(0, steps_per_epoch - resume_step_in_epoch)
                epoch_iterator = tqdm(epoch_dataloader, total=remaining)
            else:
                epoch_iterator = tqdm(epoch_dataloader)
        for data in epoch_iterator:
            stop_after_step = False
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                if dataset.load_from_cache:
                    loss = model({}, inputs=data)
                else:
                    loss = model(data)
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()

                global_step += 1
                running_loss += loss.item()

                # Logging
                if global_step % log_steps == 0:
                    avg_loss = running_loss / log_steps
                    if accelerator.is_main_process:
                        if max_steps is not None:
                            progress_bar.set_postfix({"loss": f"{avg_loss:.4f}", "step": global_step})
                            progress_bar.update(log_steps)
                        else:
                            print(f"Step {global_step} | Loss: {avg_loss:.4f}")
                        # Wandb logging
                        if use_wandb:
                            unwrapped = accelerator.unwrap_model(model)
                            wandb_log = {
                                "train/loss": avg_loss,
                                "train/learning_rate": scheduler.get_last_lr()[0],
                                "train/epoch": epoch_id,
                            }
                            wandb_log.update(_summarize_trainable_grad_norms(unwrapped))
                            wandb_log.update(_summarize_action_controller_gates(unwrapped))
                            wandb.log(wandb_log, step=global_step)
                    running_loss = 0.0

                # Save checkpoint
                model_logger.on_step_end(
                    accelerator,
                    model,
                    save_steps,
                    epoch_id=epoch_id,
                    optimizer=optimizer,
                    scheduler=scheduler,
                )

                # Check max_steps
                if max_steps is not None and global_step >= max_steps:
                    should_stop = True
                    stop_after_step = True
            _run_periodic_eval(accelerator, model, args, global_step, shard_by_rank=True)
            if stop_after_step:
                break
        if resume_step_in_epoch and epoch_id == start_epoch:
            resume_step_in_epoch = 0

        # Save at end of epoch (only if not using max_steps mode)
        if save_steps is None and max_steps is None:
            model_logger.on_epoch_end(
                accelerator,
                model,
                epoch_id,
                optimizer=optimizer,
                scheduler=scheduler,
            )

    if max_steps is not None:
        progress_bar.close()

    # Finish wandb
    if use_wandb and accelerator.is_main_process:
        wandb.finish()

    model_logger.on_training_end(
        accelerator,
        model,
        save_steps,
        epoch_id=last_epoch,
        optimizer=optimizer,
        scheduler=scheduler,
    )


def launch_data_process_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    num_workers: int = 8,
    args = None,
):
    if args is not None and getattr(args, "seed", None) is not None:
        set_seed(args.seed)
    if args is not None:
        num_workers = args.dataset_num_workers
        
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)
    model, dataloader = accelerator.prepare(model, dataloader)
    
    for data_id, data in enumerate(tqdm(dataloader)):
        with accelerator.accumulate(model):
            with torch.no_grad():
                folder = os.path.join(model_logger.output_path, str(accelerator.process_index))
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(model_logger.output_path, str(accelerator.process_index), f"{data_id}.pth")
                data = model(data)
                torch.save(data, save_path)
