#!/usr/bin/env python3
"""
Inference script for Wan2.2-TI2V-5B with trained LoRA.

Usage:
    python inference_ti2v_lora.py \
        --lora_path ./models/train/Wan2.2-TI2V-5B_drone_lora/step-10000.safetensors \
        --input_image /path/to/image.jpg \
        --prompt "your prompt here" \
        --output output.mp4
"""

import torch
import argparse
from PIL import Image
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.core.data.operators import ImageCropAndResize
from diffsynth.utils.data import save_video


def main():
    parser = argparse.ArgumentParser(description="Inference with Wan2.2-TI2V-5B + LoRA")
    parser.add_argument("--lora_path", type=str, required=True, help="Path to LoRA checkpoint")
    parser.add_argument("--lora_alpha", type=float, default=1.0, help="LoRA alpha (strength)")
    parser.add_argument("--input_image", type=str, required=True, help="Path to input image")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--output", type=str, default="output.mp4", help="Output video path")
    parser.add_argument("--height", type=int, default=480, help="Video height")
    parser.add_argument("--width", type=int, default=832, help="Video width")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--cfg_scale", type=float, default=5.0, help="CFG scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fps", type=int, default=15, help="Output video FPS")
    args = parser.parse_args()

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

    # Load LoRA
    print(f"Loading LoRA from {args.lora_path} with alpha={args.lora_alpha}...")
    pipe.load_lora(pipe.dit, args.lora_path, alpha=args.lora_alpha)

    image_processor = ImageCropAndResize(height=args.height, width=args.width, height_division_factor=16, width_division_factor=16)

    # Load input image
    print(f"Loading input image from {args.input_image}...")
    input_image = Image.open(args.input_image).convert("RGB")
    # Match training's center-crop + resize behavior.
    input_image = image_processor(input_image)

    # Generate video
    print(f"Generating video with prompt: {args.prompt}")
    video = pipe(
        prompt=args.prompt,
        input_image=input_image,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        cfg_scale=args.cfg_scale,
        seed=args.seed,
        tiled=True,
    )

    # Save video
    print(f"Saving video to {args.output}...")
    save_video(video, args.output, fps=args.fps, quality=5)
    print("Done!")


if __name__ == "__main__":
    main()
