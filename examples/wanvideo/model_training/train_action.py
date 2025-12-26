import argparse
import json
import os
import warnings

import accelerate
import torch
from safetensors.torch import load_file as load_safetensors

from diffsynth.core import UnifiedDataset
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.models.wan_video_action_controller import WanActionController, ACTION_CONTROLLER_CONFIGS
from diffsynth.diffusion import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"


ACTION_MAGNITUDE_SCALE = torch.tensor([100, 100, 100, 100, 1, 1, 1, 1], dtype=torch.float32)


class WanActionTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None,
        model_id_with_origin_paths=None,
        model_dir=None,
        tokenizer_path=None,
        trainable_models=None,
        lora_base_model=None,
        lora_target_modules="",
        lora_rank=32,
        lora_checkpoint=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        device="cpu",
        task="sft",
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        action_controller_config="wan_ti2v_5b",
        action_injection_type=None,
        action_initial_scale=None,
        action_controller_checkpoint=None,
    ):
        super().__init__()

        if not use_gradient_checkpointing:
            warnings.warn("Forcing gradient checkpointing to be enabled.")
            use_gradient_checkpointing = True

        model_configs = self.parse_model_configs(
            model_paths,
            model_id_with_origin_paths,
            fp8_models=fp8_models,
            offload_models=offload_models,
            device=device,
            model_dir=model_dir,
        )
        tokenizer_config = ModelConfig(
            model_id="Wan-AI/Wan2.2-TI2V-5B",
            origin_file_pattern="google/umt5-xxl/"
        ) if tokenizer_path is None else ModelConfig(tokenizer_path)

        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device,
            model_configs=model_configs,
            tokenizer_config=tokenizer_config
        )
        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)

        if isinstance(action_controller_config, str):
            ac_config = ACTION_CONTROLLER_CONFIGS[action_controller_config].copy()
        else:
            ac_config = action_controller_config.copy()
        if action_injection_type is not None:
            ac_config["injection_type"] = action_injection_type
        if action_initial_scale is not None:
            ac_config["initial_scale"] = action_initial_scale
        if "action_text_dim" not in ac_config and getattr(self.pipe, "text_encoder", None) is not None:
            ac_config["action_text_dim"] = self.pipe.text_encoder.dim

        self.pipe.action_controller = WanActionController(**ac_config)
        self.pipe.action_controller = self.pipe.action_controller.to(
            device=device,
            dtype=torch.bfloat16
        )

        if action_controller_checkpoint is not None:
            if action_controller_checkpoint.endswith(".safetensors"):
                state_dict = load_safetensors(action_controller_checkpoint)
            else:
                state_dict = torch.load(action_controller_checkpoint, map_location="cpu")
            self.pipe.action_controller.load_state_dict(state_dict, strict=False)
            print(f"Loaded action controller checkpoint: {action_controller_checkpoint}")

        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint,
            task=task,
        )

        self.lora_base_model = lora_base_model
        trainable_model_names = set(trainable_models.split(",")) if trainable_models else set()
        if lora_base_model not in (None, "") and not task.endswith(":data_process"):
            base_model = getattr(self.pipe, lora_base_model, None)
            if base_model is None:
                print(f"No {lora_base_model} model in the pipeline for LoRA.")
            else:
                base_model.train()
                lora_params = 0
                for name, param in base_model.named_parameters():
                    if "lora_" in name:
                        param.requires_grad = True
                        lora_params += param.numel()
                    elif lora_base_model not in trainable_model_names:
                        param.requires_grad = False
                print(f"LoRA trainable params ({lora_base_model}): {lora_params}")

        if trainable_models is not None and "action_controller" in trainable_models.split(","):
            self.pipe.action_controller.train()
            self.pipe.action_controller.requires_grad_(True)
            print(f"Action controller trainable params: {sum(p.numel() for p in self.pipe.action_controller.parameters() if p.requires_grad)}")
        else:
            self.pipe.action_controller.eval()
            self.pipe.action_controller.requires_grad_(False)

        self.action_magnitude_scale = ACTION_MAGNITUDE_SCALE
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task = task
        self.task_to_loss = {
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
        }
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary

    def parse_action(self, action_str_or_tensor):
        if action_str_or_tensor is None:
            return None

        if isinstance(action_str_or_tensor, str):
            action = json.loads(action_str_or_tensor)
            action = torch.tensor(action, dtype=torch.float32)
        elif isinstance(action_str_or_tensor, list):
            action = torch.tensor(action_str_or_tensor, dtype=torch.float32)
        else:
            action = action_str_or_tensor

        if len(action.shape) == 4 and action.shape[2] == 2:
            action_magnitude = action[:, :, 1, :]
        elif len(action.shape) == 3 and action.shape[1] == 2 and action.shape[2] == 8:
            action_magnitude = action.unsqueeze(0)[:, :, 1, :]
        elif len(action.shape) == 2:
            action_magnitude = action.unsqueeze(0)
        else:
            action_magnitude = action

        action_magnitude = action_magnitude * self.action_magnitude_scale.to(action_magnitude.device, action_magnitude.dtype)

        return action_magnitude

    def get_pipeline_inputs(self, data):
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        inputs_shared = {
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }

        if "action" in data:
            action_magnitude = self.parse_action(data["action"])
            if action_magnitude is not None:
                inputs_shared["action_magnitude"] = action_magnitude

        inputs_shared["input_image"] = data["video"][0]

        return inputs_shared, inputs_posi, inputs_nega

    def forward(self, data, inputs=None):
        if inputs is None:
            inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)

        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)

        loss = self.task_to_loss[self.task](self.pipe, *inputs)
        return loss

    def export_trainable_state_dict(self, state_dict, remove_prefix=None):
        trainable_param_names = self.trainable_param_names()
        state_dict = {name: param for name, param in state_dict.items() if name in trainable_param_names}
        if self.lora_base_model not in (None, ""):
            prefix = f"pipe.{self.lora_base_model}."
            state_dict = {k: v for k, v in state_dict.items() if not (k.startswith(prefix) and "lora_" in k)}

        if remove_prefix is not None:
            state_dict_ = {}
            for name, param in state_dict.items():
                if name.startswith("pipe.action_controller."):
                    name = name[len("pipe.action_controller."):]
                elif remove_prefix and name.startswith(remove_prefix):
                    name = name[len(remove_prefix):]
                state_dict_[name] = param
            state_dict = state_dict_
        return state_dict


def action_parser():
    parser = argparse.ArgumentParser(description="Wan Video + Action Controller Training")
    parser = add_general_config(parser)
    parser = add_video_size_config(parser)
    parser.set_defaults(data_file_keys="video")
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--action_controller_config", type=str, default="wan_ti2v_5b",
                        choices=list(ACTION_CONTROLLER_CONFIGS.keys()))
    parser.add_argument("--action_injection_type", type=str, default=None,
                        choices=["patch_add", "layer_add"])
    parser.add_argument("--action_initial_scale", type=float, default=None)
    parser.add_argument("--action_controller_checkpoint", type=str, default=None)
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0)
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0)
    parser.add_argument("--initialize_model_on_cpu", default=False, action="store_true")
    return parser


if __name__ == "__main__":
    parser = action_parser()
    args = parser.parse_args()
    args.remove_prefix_in_ckpt = "pipe.action_controller."

    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=True)],
    )

    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=("video",),
        main_data_operator=UnifiedDataset.default_video_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
            num_frames=args.num_frames,
            time_division_factor=4,
            time_division_remainder=1,
        ),
    )

    model = WanActionTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        model_dir=args.model_dir,
        tokenizer_path=args.tokenizer_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        task=args.task,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        action_controller_config=args.action_controller_config,
        action_injection_type=args.action_injection_type,
        action_initial_scale=args.action_initial_scale,
        action_controller_checkpoint=args.action_controller_checkpoint,
    )

    action_cfg = ACTION_CONTROLLER_CONFIGS[args.action_controller_config].copy()
    if args.action_injection_type is not None:
        action_cfg["injection_type"] = args.action_injection_type
    if args.action_initial_scale is not None:
        action_cfg["initial_scale"] = args.action_initial_scale
    if "action_text_dim" not in action_cfg and getattr(model.pipe, "text_encoder", None) is not None:
        action_cfg["action_text_dim"] = model.pipe.text_encoder.dim
    base_model_id = "Wan-AI/Wan2.2-TI2V-5B" if args.model_dir is None else None
    save_config = {
        "base_model_id": base_model_id,
        "base_model_dir": args.model_dir,
        "action_controller_config": action_cfg,
        "action_magnitude_scale": ACTION_MAGNITUDE_SCALE.tolist(),
        "lora": {
            "lora_base_model": args.lora_base_model,
            "lora_target_modules": args.lora_target_modules,
            "lora_rank": args.lora_rank,
        },
    }

    checkpoint_metadata = vars(args).copy()

    def save_lora_weights(checkpoint_dir, accelerator, model):
        base_name = args.lora_base_model
        if base_name in (None, ""):
            return
        state_dict = accelerator.get_state_dict(model)
        prefix = f"pipe.{base_name}."
        lora_state = {
            k[len(prefix):]: v
            for k, v in state_dict.items()
            if k.startswith(prefix) and "lora_" in k
        }
        if not lora_state:
            return
        path = os.path.join(checkpoint_dir, "lora.safetensors")
        accelerator.save(lora_state, path, safe_serialization=True)

    def save_action_train_config(checkpoint_dir, accelerator, model):
        if not accelerator.is_main_process:
            return
        path = os.path.join(checkpoint_dir, "action_train_config.json")
        with open(path, "w") as f:
            json.dump(save_config, f, indent=2)
        save_lora_weights(checkpoint_dir, accelerator, model)

    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        save_full_state=args.save_full_state,
        max_checkpoints=args.max_checkpoints,
        checkpoint_metadata=checkpoint_metadata,
        extra_checkpoint_saver=save_action_train_config,
        save_root_steps=False,
        checkpoint_weights_name="action_controller.safetensors",
    )
    os.makedirs(args.output_path, exist_ok=True)
    with open(os.path.join(args.output_path, "action_train_config.json"), "w") as f:
        json.dump(save_config, f, indent=2)

    launch_training_task(accelerator, dataset, model, model_logger, args=args)
