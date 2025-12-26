#!/usr/bin/env python3
"""
Batch inference script for Wan2.2-TI2V-5B with trained LoRA.
Supports CSV/JSON input, random sampling, and streaming generation.

Usage:
    # From CSV (randomly sample 10 examples)
    python batch_inference_ti2v.py \
        --lora_path ./models/train/Wan2.2-TI2V-5B_drone_lora/step-10000.safetensors \
        --input_csv /path/to/metadata.csv \
        --base_path /path/to/dataset \
        --num_samples 10 \
        --output_dir ./inference_results

    # From JSON file
    python batch_inference_ti2v.py \
        --lora_path ./models/train/xxx.safetensors \
        --input_json examples.json \
        --output_dir ./inference_results

    # From local base model dir
    python batch_inference_ti2v.py \
        --model_dir /path/to/Wan2.1-T2V-1.3B \
        --force_i2v_fuse \
        --lora_path ./models/train/xxx.safetensors \
        --input_csv /path/to/metadata.csv \
        --output_dir ./inference_results

    # Compare with/without LoRA
    python batch_inference_ti2v.py \
        --lora_path ./models/train/xxx.safetensors \
        --input_csv /path/to/metadata.csv \
        --base_path /path/to/dataset \
        --num_samples 5 \
        --compare_baseline \
        --output_dir ./inference_results
"""

import torch
import argparse
import json
import random
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from diffsynth.core import ModelConfig, load_model_dir
from diffsynth.pipelines.wan_video import WanVideoPipeline
from diffsynth.core.data.operators import ImageCropAndResize
from diffsynth.utils.data import save_video


def load_examples_from_csv(csv_path: str, base_path: str, num_samples: int = 10, seed: int = 42):
    """Load and sample examples from CSV file."""
    df = pd.read_csv(csv_path)

    # Random sample
    if num_samples and num_samples < len(df):
        random.seed(seed)
        indices = random.sample(range(len(df)), num_samples)
        df = df.iloc[indices]

    examples = []
    base_path = Path(base_path) if base_path else Path(csv_path).parent

    for _, row in df.iterrows():
        example = {
            "video": str(base_path / row["video"]) if not str(row["video"]).startswith("/") else row["video"],
            "prompt": row["prompt"],
            "input_image": str(base_path / row["input_image"]) if not str(row["input_image"]).startswith("/") else row["input_image"],
        }
        examples.append(example)

    return examples


def load_examples_from_json(json_path: str):
    """Load examples from JSON file."""
    with open(json_path, "r") as f:
        examples = json.load(f)
    return examples


def main():
    parser = argparse.ArgumentParser(description="Batch inference with Wan2.2-TI2V-5B + LoRA")

    # Input options
    parser.add_argument("--input_csv", type=str, help="Path to input CSV file")
    parser.add_argument("--input_json", type=str, help="Path to input JSON file")
    parser.add_argument("--base_path", type=str, default="", help="Base path for relative paths in CSV")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")

    # Model options
    parser.add_argument("--lora_path", type=str, required=True, help="Path to LoRA checkpoint")
    parser.add_argument("--lora_alpha", type=float, default=1.0, help="LoRA alpha (strength)")
    parser.add_argument("--compare_baseline", action="store_true", help="Also generate without LoRA for comparison")
    parser.add_argument("--model_dir", type=str, default=None, help="Local base model directory (diffsynth model index).")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Optional local tokenizer path.")
    parser.add_argument(
        "--force_i2v_fuse",
        action="store_true",
        help="Force fuse_vae_embedding_in_latents and disable VAE embedding channels (I2V style).",
    )

    # Generation options
    parser.add_argument("--height", type=int, default=480, help="Video height")
    parser.add_argument("--width", type=int, default=832, help="Video width")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--cfg_scale", type=float, default=5.0, help="CFG scale")
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        help="Negative prompt for better stability",
    )
    parser.add_argument("--fps", type=int, default=15, help="Output video FPS")

    # Output options
    parser.add_argument("--output_dir", type=str, default="./inference_results", help="Output directory")

    args = parser.parse_args()

    # Load examples
    if args.input_csv:
        examples = load_examples_from_csv(args.input_csv, args.base_path, args.num_samples, args.seed)
    elif args.input_json:
        examples = load_examples_from_json(args.input_json)
    else:
        raise ValueError("Must provide either --input_csv or --input_json")

    print(f"Loaded {len(examples)} examples")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save examples info
    with open(output_dir / "examples.json", "w") as f:
        json.dump(examples, f, indent=2)

    if args.model_dir:
        print(f"Loading base model from {args.model_dir}...")
        _, components = load_model_dir(args.model_dir)
        model_configs = [ModelConfig(path=path) for path in components.values()]
        if args.tokenizer_path:
            tokenizer_config = ModelConfig(path=args.tokenizer_path)
        else:
            tokenizer_config = ModelConfig(
                model_id="Wan-AI/Wan2.1-T2V-1.3B",
                origin_file_pattern="google/umt5-xxl/",
            )
        pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cuda",
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
            redirect_common_files=False,
        )
    else:
        print("Loading Wan2.2-TI2V-5B pipeline...")
        pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cuda",
            model_configs=[
                ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
                ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
                ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="Wan2.2_VAE.pth"),
            ],
        )
    if args.force_i2v_fuse:
        pipe.dit.fuse_vae_embedding_in_latents = True
        pipe.dit.require_vae_embedding = False
        print("Enabled fuse_vae_embedding_in_latents and disabled VAE embedding (force_i2v_fuse).")

    image_processor = ImageCropAndResize(height=args.height, width=args.width, height_division_factor=16, width_division_factor=16)

    def generate_videos(pipe, examples, output_dir, suffix, desc, load_lora=False):
        if load_lora and not getattr(pipe, "_lora_loaded", False):
            print(f"Loading LoRA from {args.lora_path}...")
            pipe.load_lora(pipe.dit, args.lora_path, alpha=args.lora_alpha)
            pipe._lora_loaded = True

        for i, example in enumerate(tqdm(examples, desc=desc)):
            input_image_path = example["input_image"]
            prompt = example["prompt"]

            # Load and resize input image
            input_image = Image.open(input_image_path).convert("RGB")
            # Match training's center-crop + resize behavior.
            input_image = image_processor(input_image)

            seed = args.seed + i  # Different seed for each example

            print(f"\n[{i+1}/{len(examples)}] Generating {suffix}...")
            print(f"  Prompt: {prompt[:100]}...")

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
                tiled=True,
            )

            output_path = output_dir / f"{i:03d}_{suffix}.mp4"
            save_video(video, str(output_path), fps=args.fps, quality=5)
            print(f"  Saved: {output_path}")

            if suffix == "lora":
                input_image.save(output_dir / f"{i:03d}_input.jpg")

    if args.compare_baseline:
        print("Generating baseline videos (no LoRA)...")
        generate_videos(pipe, examples, output_dir, "baseline", "Generating baseline", load_lora=False)

    print("Generating LoRA videos...")
    generate_videos(pipe, examples, output_dir, "lora", "Generating LoRA", load_lora=True)

    print(f"\nDone! Results saved to {output_dir}")
    print(f"  - {len(examples)} videos generated")
    if args.compare_baseline:
        print(f"  - Baseline comparison videos also generated")


if __name__ == "__main__":
    main()
