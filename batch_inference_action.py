#!/usr/bin/env python3
"""
Batch inference script for Wan action controller.

Inputs: CSV with columns like:
  - video: path to a source clip (used to grab the first frame as input_image)
  - prompt: text prompt
  - action: JSON string of shape [T][2][8] (binary + magnitude) or [T][8] (magnitude only)

Outputs: generated videos with action overlay.
"""

import argparse
import ast
import json
import math
import os
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from diffsynth.core import ModelConfig, load_model_dir
from diffsynth.core.data.operators import ImageCropAndResize
from diffsynth.models.wan_video_action_controller import ACTION_CONTROLLER_CONFIGS, WanActionController
from diffsynth.pipelines.wan_video import WanVideoPipeline
from diffsynth.utils.data import save_video


ACTION_NAMES = [
    "Forward",
    "Back",
    "Left",
    "Right",
    "Yaw Left",
    "Yaw Right",
    "Pitch Up",
    "Pitch Down",
]
ACTION_TRANSLATION = ACTION_NAMES[:4]
ACTION_ROTATION = ACTION_NAMES[4:]
DEFAULT_ACTION_MAGNITUDE_SCALE = torch.tensor([100, 100, 100, 100, 1, 1, 1, 1], dtype=torch.float32)

T5_TRANSLATION_PHRASES = [
    "Camera remains still.",
    "Camera moves forward.",
    "Camera moves backward.",
    "Camera moves left.",
    "Camera moves right.",
    "Camera moves forward and left.",
    "Camera moves forward and right.",
    "Camera moves backward and left.",
    "Camera moves backward and right.",
]
T5_ROTATION_PHRASES = [
    "Camera remains still.",
    "Camera turns left.",
    "Camera turns right.",
    "Camera tilts up.",
    "Camera tilts down.",
    "Camera turns left and tilts up.",
    "Camera turns left and tilts down.",
    "Camera turns right and tilts up.",
    "Camera turns right and tilts down.",
]


def parse_action_value(action_value):
    if action_value is None:
        return None
    if isinstance(action_value, float) and np.isnan(action_value):
        return None
    if isinstance(action_value, (list, tuple)):
        action = action_value
    else:
        raw = str(action_value).strip()
        if not raw:
            return None
        try:
            action = json.loads(raw)
        except json.JSONDecodeError:
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
        raise ValueError(f"Unexpected action shape: {tuple(action.shape)} (expected [T, 8] or [B, T, 8])")
    return action_magnitude


def expected_action_length(num_frames, time_division_factor, time_division_remainder):
    if num_frames is None:
        return None
    if num_frames < time_division_remainder:
        return None
    return max(0, (num_frames - time_division_remainder) // time_division_factor)


def align_action_length(action_magnitude, target_len):
    if target_len is None:
        return action_magnitude
    current_len = action_magnitude.shape[1]
    if current_len == target_len:
        return action_magnitude
    if current_len > target_len:
        return action_magnitude[:, :target_len]
    pad = target_len - current_len
    pad_shape = (action_magnitude.shape[0], pad, action_magnitude.shape[2])
    action_magnitude = torch.cat([action_magnitude, torch.zeros(pad_shape, dtype=action_magnitude.dtype)], dim=1)
    return action_magnitude


def load_first_frame(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Failed to read first frame: {video_path}")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame)


def resolve_path(base_path, raw_path):
    if not raw_path:
        return None
    raw_path = str(raw_path)
    if raw_path.startswith("/"):
        return Path(raw_path)
    return Path(base_path) / raw_path


def action_steps_to_sets(action_magnitude, threshold=1e-6):
    action_sets = []
    for step in action_magnitude:
        active = {ACTION_NAMES[i] for i, value in enumerate(step.tolist()) if value > threshold}
        action_sets.append(active)
    return action_sets


def build_frame_actions(action_sets, num_frames, stride=None, offset=None):
    if not action_sets:
        return [set() for _ in range(num_frames)]
    num_steps = len(action_sets)
    if offset is None:
        if (num_frames - 1) % num_steps == 0:
            offset = 1
        else:
            offset = 0
    if stride is None:
        if offset and (num_frames - offset) % num_steps == 0:
            stride = (num_frames - offset) // num_steps
        elif num_frames % num_steps == 0:
            stride = num_frames // num_steps
        else:
            stride = max(1, (num_frames - offset) // num_steps)
    remainder = max(0, num_frames - offset - stride * num_steps)
    frame_actions = [set() for _ in range(num_frames)]
    start = offset
    for idx, actions in enumerate(action_sets):
        seg_len = stride + (1 if idx < remainder else 0)
        end = min(num_frames, start + seg_len)
        for frame_id in range(start, end):
            frame_actions[frame_id].update(actions)
        start = end
    return frame_actions


def _action_to_phrase_indices(action_magnitude: torch.Tensor, threshold: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    if action_magnitude.dim() == 3:
        if action_magnitude.shape[0] != 1:
            raise ValueError(f"Expected action_magnitude shape [T,8] or [1,T,8], got {tuple(action_magnitude.shape)}")
        action_magnitude = action_magnitude[0]
    if action_magnitude.dim() != 2 or action_magnitude.shape[-1] != 8:
        raise ValueError(f"Expected action_magnitude shape [T,8], got {tuple(action_magnitude.shape)}")

    forward = action_magnitude[..., 0] > threshold
    back = action_magnitude[..., 1] > threshold
    left = action_magnitude[..., 2] > threshold
    right = action_magnitude[..., 3] > threshold
    yaw_left = action_magnitude[..., 4] > threshold
    yaw_right = action_magnitude[..., 5] > threshold
    pitch_up = action_magnitude[..., 6] > threshold
    pitch_down = action_magnitude[..., 7] > threshold

    fb_conflict = forward & back
    forward = forward & ~fb_conflict
    back = back & ~fb_conflict
    lr_conflict = left & right
    left = left & ~lr_conflict
    right = right & ~lr_conflict
    yaw_conflict = yaw_left & yaw_right
    yaw_left = yaw_left & ~yaw_conflict
    yaw_right = yaw_right & ~yaw_conflict
    pitch_conflict = pitch_up & pitch_down
    pitch_up = pitch_up & ~pitch_conflict
    pitch_down = pitch_down & ~pitch_conflict

    trans_idx = torch.zeros_like(forward, dtype=torch.long)
    trans_idx[forward & left] = 5
    trans_idx[forward & right] = 6
    trans_idx[back & left] = 7
    trans_idx[back & right] = 8
    trans_idx[forward & ~left & ~right] = 1
    trans_idx[back & ~left & ~right] = 2
    trans_idx[left & ~forward & ~back] = 3
    trans_idx[right & ~forward & ~back] = 4

    rot_idx = torch.zeros_like(yaw_left, dtype=torch.long)
    rot_idx[yaw_left & pitch_up] = 5
    rot_idx[yaw_left & pitch_down] = 6
    rot_idx[yaw_right & pitch_up] = 7
    rot_idx[yaw_right & pitch_down] = 8
    rot_idx[yaw_left & ~pitch_up & ~pitch_down] = 1
    rot_idx[yaw_right & ~pitch_up & ~pitch_down] = 2
    rot_idx[pitch_up & ~yaw_left & ~yaw_right] = 3
    rot_idx[pitch_down & ~yaw_left & ~yaw_right] = 4

    return trans_idx, rot_idx


def build_frame_step_ids(num_steps: int, num_frames: int, stride: int | None = None, offset: int | None = None) -> list[int | None]:
    if num_steps <= 0:
        return [None for _ in range(num_frames)]
    if offset is None:
        if (num_frames - 1) % num_steps == 0:
            offset = 1
        else:
            offset = 0
    if stride is None:
        if offset and (num_frames - offset) % num_steps == 0:
            stride = (num_frames - offset) // num_steps
        elif num_frames % num_steps == 0:
            stride = num_frames // num_steps
        else:
            stride = max(1, (num_frames - offset) // num_steps)
    remainder = max(0, num_frames - offset - stride * num_steps)
    frame_steps: list[int | None] = [None for _ in range(num_frames)]
    start = offset
    for step_id in range(num_steps):
        seg_len = stride + (1 if step_id < remainder else 0)
        end = min(num_frames, start + seg_len)
        for frame_id in range(start, end):
            frame_steps[frame_id] = step_id
        start = end
    return frame_steps


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
        if active:
            color = (255, 255, 255)
            thickness = 3
        else:
            color = (60, 60, 60)
            thickness = 2
        cv2.arrowedLine(frame, start_pt, end_pt, color, thickness, tipLength=0.4)


def draw_rotation_indicator(frame, position, size, active, direction):
    x, y = position
    dx, dy = direction
    arrow_half_len = size // 3
    start_pt = (x - dx * arrow_half_len, y - dy * arrow_half_len)
    end_pt = (x + dx * arrow_half_len, y + dy * arrow_half_len)
    if active:
        color = (255, 255, 255)
        thickness = 3
    else:
        color = (40, 40, 40)
        thickness = 2
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


def load_action_controller(action_config_path, action_ckpt_path, device, torch_dtype, overrides=None):
    overrides = overrides or {}
    action_cfg = None
    action_scale = None
    if action_config_path:
        with open(action_config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        action_cfg = cfg.get("action_controller_config")
        action_scale = cfg.get("action_magnitude_scale")
    if action_cfg is None:
        action_cfg = overrides.get("action_controller_config", "wan_ti2v_5b")
    if isinstance(action_cfg, str):
        action_cfg = ACTION_CONTROLLER_CONFIGS[action_cfg].copy()
    else:
        action_cfg = action_cfg.copy()
    if overrides.get("action_injection_type") is not None:
        action_cfg["injection_type"] = overrides["action_injection_type"]
    if overrides.get("action_initial_scale") is not None:
        action_cfg["initial_scale"] = overrides["action_initial_scale"]
    action_controller = WanActionController(**action_cfg).to(device=device, dtype=torch_dtype)
    if action_ckpt_path:
        if str(action_ckpt_path).endswith(".safetensors"):
            from safetensors.torch import load_file as load_safetensors
            state_dict = load_safetensors(action_ckpt_path)
        else:
            state_dict = torch.load(action_ckpt_path, map_location="cpu")
        action_controller.load_state_dict(state_dict, strict=False)
    if action_scale is None:
        action_scale = DEFAULT_ACTION_MAGNITUDE_SCALE.tolist()
    return action_controller, torch.tensor(action_scale, dtype=torch.float32)


def _select_latest_checkpoint(checkpoints_dir):
    if not checkpoints_dir or not os.path.isdir(checkpoints_dir):
        return None
    candidates = []
    for name in os.listdir(checkpoints_dir):
        if not name.startswith("step-"):
            continue
        step_str = name.replace("step-", "")
        if not step_str.isdigit():
            continue
        step = int(step_str)
        candidates.append((step, os.path.join(checkpoints_dir, name)))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def resolve_action_assets(action_dir, action_controller_path, action_config_path):
    action_dir = Path(action_dir) if action_dir else None
    lora_path = None
    if action_dir:
        if action_config_path is None:
            candidate = action_dir / "action_train_config.json"
            if candidate.exists():
                action_config_path = str(candidate)
        if action_controller_path is None:
            candidate = action_dir / "action_controller.safetensors"
            if candidate.exists():
                action_controller_path = str(candidate)
        candidate = action_dir / "lora.safetensors"
        if candidate.exists():
            lora_path = str(candidate)

        if action_controller_path is None:
            latest_ckpt = _select_latest_checkpoint(action_dir / "checkpoints")
            if latest_ckpt:
                latest_ckpt = Path(latest_ckpt)
                candidate = latest_ckpt / "action_controller.safetensors"
                if candidate.exists():
                    action_controller_path = str(candidate)
                if action_config_path is None:
                    candidate = latest_ckpt / "action_train_config.json"
                    if candidate.exists():
                        action_config_path = str(candidate)
                candidate = latest_ckpt / "lora.safetensors"
                if candidate.exists():
                    lora_path = str(candidate)
    if lora_path is None and action_controller_path:
        candidate = Path(action_controller_path).parent / "lora.safetensors"
        if candidate.exists():
            lora_path = str(candidate)
    return action_controller_path, action_config_path, lora_path


def load_examples_from_csv(csv_path, base_path, num_samples, seed, video_column, prompt_column, action_column, input_image_column):
    df = pd.read_csv(csv_path)
    if num_samples is not None and num_samples < len(df):
        random.seed(seed)
        df = df.sample(n=num_samples, random_state=seed)
    base_path = Path(base_path) if base_path else Path(csv_path).parent
    examples = []
    for _, row in df.iterrows():
        video_path = resolve_path(base_path, row.get(video_column))
        prompt = row.get(prompt_column)
        action_value = row.get(action_column)
        input_image_path = None
        if input_image_column and input_image_column in row:
            input_image_path = resolve_path(base_path, row.get(input_image_column))
        examples.append(
            {
                "video_path": video_path,
                "prompt": prompt,
                "action": action_value,
                "input_image_path": input_image_path,
            }
        )
    return examples


def main():
    parser = argparse.ArgumentParser(description="Batch inference with Wan action controller.")
    parser.add_argument("--input_csv", type=str, required=True, help="CSV with video/prompt/action columns.")
    parser.add_argument("--base_path", type=str, default="", help="Base path for relative CSV paths.")
    parser.add_argument("--num_samples", type=int, default=None, help="Randomly sample N rows.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling and generation.")

    parser.add_argument("--video_column", type=str, default="video", help="Column name for video path.")
    parser.add_argument("--prompt_column", type=str, default="prompt", help="Column name for prompt.")
    parser.add_argument("--action_column", type=str, default="action", help="Column name for action sequence.")
    parser.add_argument("--input_image_column", type=str, default="", help="Optional column for input image path.")

    parser.add_argument("--model_dir", type=str, required=True, help="Base model directory with model_index.json.")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Optional local tokenizer path.")

    parser.add_argument("--action_dir", type=str, default=None, help="Action training output dir.")
    parser.add_argument("--action_controller_path", type=str, default=None, help="Path to action_controller.safetensors.")
    parser.add_argument("--action_config_path", type=str, default=None, help="Path to action_train_config.json.")
    parser.add_argument("--action_injection_type", type=str, default=None, help="Override action injection type.")
    parser.add_argument("--action_initial_scale", type=float, default=None, help="Override action initial scale.")

    parser.add_argument("--height", type=int, default=480, help="Video height.")
    parser.add_argument("--width", type=int, default=832, help="Video width.")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames.")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Inference steps.")
    parser.add_argument("--cfg_scale", type=float, default=5.0, help="CFG scale.")
    parser.add_argument("--negative_prompt", type=str, default="", help="Negative prompt.")
    parser.add_argument("--fps", type=int, default=15, help="Output FPS.")

    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference.")
    parser.add_argument("--dtype", type=str, default="bf16", help="torch dtype: bf16/fp16/fp32.")

    parser.add_argument("--time_division_factor", type=int, default=4, help="Temporal downsample factor.")
    parser.add_argument("--time_division_remainder", type=int, default=1, help="Temporal remainder.")
    parser.add_argument("--action_stride", type=int, default=None, help="Stride (frames per action step) for overlay.")
    parser.add_argument("--action_offset", type=int, default=None, help="Frame offset for overlay.")

    parser.add_argument("--output_dir", type=str, default="./inference_action", help="Output directory.")
    parser.add_argument("--save_raw", action="store_true", help="Also save raw (no overlay) video.")
    parser.add_argument("--log_action_text", action="store_true",
                        help="Write a JSON log showing per-step and per-frame action->(translation/rotation) text mapping used for T5 embedding.")
    parser.add_argument("--action_text_threshold", type=float, default=1e-6,
                        help="Threshold for action->text mapping (should match pipeline).")

    args = parser.parse_args()

    input_image_column = args.input_image_column or None
    examples = load_examples_from_csv(
        args.input_csv,
        args.base_path,
        args.num_samples,
        args.seed,
        args.video_column,
        args.prompt_column,
        args.action_column,
        input_image_column,
    )
    if not examples:
        raise RuntimeError("No samples loaded from CSV.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _, components = load_model_dir(args.model_dir)
    model_configs = [ModelConfig(path=path) for path in components.values()]
    if args.tokenizer_path:
        tokenizer_config = ModelConfig(path=args.tokenizer_path)
    else:
        tokenizer_config = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/")

    torch_dtype = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }.get(args.dtype.lower(), torch.bfloat16)

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch_dtype,
        device=args.device,
        model_configs=model_configs,
        tokenizer_config=tokenizer_config,
        audio_processor_config=None,
        redirect_common_files=False,
    )

    action_controller_path, action_config_path, lora_path = resolve_action_assets(
        args.action_dir,
        args.action_controller_path,
        args.action_config_path,
    )
    if action_controller_path is None:
        raise ValueError("Missing action controller weights. Set --action_dir or --action_controller_path.")

    action_controller, action_scale = load_action_controller(
        action_config_path,
        action_controller_path,
        device=args.device,
        torch_dtype=torch_dtype,
        overrides={
            "action_injection_type": args.action_injection_type,
            "action_initial_scale": args.action_initial_scale,
        },
    )
    pipe.action_controller = action_controller
    if lora_path:
        if getattr(pipe, "dit", None) is None:
            print("LoRA checkpoint found but pipe.dit is missing; skipping LoRA load.")
        else:
            pipe.load_lora(pipe.dit, lora_path)
            print(f"Loaded LoRA: {lora_path}")

    image_processor = ImageCropAndResize(
        height=args.height,
        width=args.width,
        height_division_factor=16,
        width_division_factor=16,
    )

    expected_len = expected_action_length(
        args.num_frames,
        args.time_division_factor,
        args.time_division_remainder,
    )

    for idx, sample in enumerate(tqdm(examples, desc="Generating", unit="sample")):
        prompt = sample["prompt"]
        video_path = sample["video_path"]
        if video_path is None or not Path(video_path).exists():
            raise FileNotFoundError(f"Video path not found: {video_path}")
        if sample["input_image_path"] and Path(sample["input_image_path"]).exists():
            input_image = Image.open(sample["input_image_path"]).convert("RGB")
        else:
            input_image = load_first_frame(video_path)
        input_image = image_processor(input_image)

        action_magnitude = parse_action_value(sample["action"])
        if action_magnitude is None:
            raise ValueError(f"Missing action for row {idx}")
        action_magnitude = align_action_length(action_magnitude, expected_len)
        action_magnitude_raw = action_magnitude.clone()

        action_magnitude = action_magnitude * action_scale.to(action_magnitude.device, action_magnitude.dtype)
        action_magnitude = action_magnitude.to(device=args.device, dtype=torch_dtype)

        seed = args.seed + idx
        with torch.no_grad():
            video = pipe(
                prompt=prompt,
                negative_prompt=args.negative_prompt,
                input_image=input_image,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                num_inference_steps=args.num_inference_steps,
                cfg_scale=args.cfg_scale,
                seed=seed,
                action_magnitude=action_magnitude,
                tiled=True,
            )

        if args.log_action_text:
            action_magnitude_scaled_cpu = (action_magnitude_raw[0] * action_scale).to(dtype=torch.float32, device="cpu")
            trans_idx, rot_idx = _action_to_phrase_indices(action_magnitude_scaled_cpu, threshold=args.action_text_threshold)
            step_logs = []
            for step_id in range(action_magnitude_raw.shape[1]):
                trans_id = int(trans_idx[step_id].item())
                rot_id = int(rot_idx[step_id].item())
                trans_phrase = T5_TRANSLATION_PHRASES[trans_id]
                rot_phrase = T5_ROTATION_PHRASES[rot_id]
                if trans_id == 0 and rot_id == 0:
                    combo_phrase = "Camera remains still."
                elif trans_id == 0:
                    combo_phrase = rot_phrase
                elif rot_id == 0:
                    combo_phrase = trans_phrase
                else:
                    combo_phrase = f"{trans_phrase} {rot_phrase}"
                step_logs.append(
                    {
                        "step_id": step_id,
                        "trans_idx": trans_id,
                        "rot_idx": rot_id,
                        "combo_idx": trans_id * 9 + rot_id,
                        "translation_phrase": trans_phrase,
                        "rotation_phrase": rot_phrase,
                        "combo_phrase": combo_phrase,
                        "action_magnitude_raw": [float(v) for v in action_magnitude_raw[0, step_id].tolist()],
                        "action_magnitude_scaled": [float(v) for v in action_magnitude_scaled_cpu[step_id].tolist()],
                    }
                )

            frame_step_ids = build_frame_step_ids(
                num_steps=len(step_logs),
                num_frames=len(video),
                stride=args.action_stride,
                offset=args.action_offset,
            )
            frame_logs = []
            for frame_id, step_id_overlay in enumerate(frame_step_ids):
                step_id_model = None
                if frame_id >= args.time_division_remainder and args.time_division_factor > 0:
                    candidate = (frame_id - args.time_division_remainder) // args.time_division_factor
                    if 0 <= candidate < len(step_logs):
                        step_id_model = int(candidate)

                overlay_phrase = None
                if step_id_overlay is not None:
                    overlay_phrase = step_logs[int(step_id_overlay)]["combo_phrase"]

                model_phrase = None
                if step_id_model is not None:
                    model_phrase = step_logs[step_id_model]["combo_phrase"]

                frame_logs.append(
                    {
                        "frame_id": frame_id,
                        "step_id_model": step_id_model,
                        "combo_phrase_model": model_phrase,
                        "step_id_overlay": int(step_id_overlay) if step_id_overlay is not None else None,
                        "combo_phrase_overlay": overlay_phrase,
                    }
                )

            stem = Path(video_path).stem
            log_path = output_dir / f"{idx:03d}_{stem}_action_text_log.json"
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "video_path": str(video_path),
                        "prompt": prompt,
                        "num_frames": len(video),
                        "time_division_factor": args.time_division_factor,
                        "time_division_remainder": args.time_division_remainder,
                        "action_stride": args.action_stride,
                        "action_offset": args.action_offset,
                        "action_text_threshold": args.action_text_threshold,
                        "translation_phrases": T5_TRANSLATION_PHRASES,
                        "rotation_phrases": T5_ROTATION_PHRASES,
                        "action_magnitude_scale": [float(v) for v in action_scale.tolist()],
                        "steps": step_logs,
                        "frames": frame_logs,
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

        action_sets = action_steps_to_sets(action_magnitude_raw[0].cpu())
        frame_actions = build_frame_actions(
            action_sets,
            num_frames=len(video),
            stride=args.action_stride,
            offset=args.action_offset,
        )
        overlay_frames = overlay_actions_on_frames(video, frame_actions)

        stem = Path(video_path).stem
        output_path = output_dir / f"{idx:03d}_{stem}_action_overlay.mp4"
        save_video(overlay_frames, str(output_path), fps=args.fps, quality=5)

        if args.save_raw:
            raw_path = output_dir / f"{idx:03d}_{stem}_raw.mp4"
            save_video(video, str(raw_path), fps=args.fps, quality=5)


if __name__ == "__main__":
    main()
