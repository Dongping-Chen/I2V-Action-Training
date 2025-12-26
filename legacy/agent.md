# Wan Video Action Control Fine-tuning 实现计划 v3

## 项目概述

为 Wan 2.2 TI2V 5B 模型添加 Action Control 能力，使其能够根据实时动作指令生成视频。

### 模型配置
- **Base Model**: Wan2.2-TI2V-5B
- **模型大小说明**: 训练/推理优先用本地 `--model_dir`（含 `model_index.json`）；`model_id_with_origin_paths` 仅作兼容。如需 DeepSpeed 再指定 accelerate config
- **Action Types**: 8种 (Forward, Back, Left, Right, Yaw Left, Yaw Right, Pitch Up, Pitch Down)
- **首帧处理**: 使用 TI2V 的 `fuse_vae_embedding_in_latents` 机制融合首帧
- **Action 数量**: 20个（对应 latent frame 1-20，frame 0 是首帧）

### 数据格式
- **视频**: 81 帧 → 21 个 latent frames
- **Action**: `(20, 2, 8)` - 20帧 × [binary, magnitude] × 8种动作
- **Magnitude 归一化**: 对前 4 个动作（Forward/Back/Left/Right）乘以 100，以对齐 Yaw/Pitch 的量级  
  `action_magnitude_scale = [100, 100, 100, 100, 1, 1, 1, 1]`
- **CSV 列**: `video`, `prompt`, `action` (action 为 JSON 字符串)

---

## 文件清单

### 新建文件
| 文件 | 用途 |
|------|------|
| `diffsynth/models/wan_video_action_controller.py` | ActionModule 核心实现 |
| `examples/wanvideo/model_training/train_action.py` | Action训练脚本 |
| `examples/wanvideo/model_training/train_action.sh` | 训练启动脚本 |

### 修改文件
| 文件 | 修改内容 |
|------|---------|
| `diffsynth/pipelines/wan_video.py` | 修改 `model_fn_wan_video` + pipeline 添加 `action_controller` + `__call__` 接收 action 输入 |

---

## 详细实现

### 1. `wan_video_action_controller.py`

```python
# diffsynth/models/wan_video_action_controller.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False


class WanRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight


class WanActionController(nn.Module):
    """
    Action Controller for Wan Video Model (Drone Control)

    Actions: Forward, Back, Left, Right, Yaw Left, Yaw Right, Pitch Up, Pitch Down

    设计:
    - 支持多种注入方式 (injection_type):
      1) cross_attn_frame: 逐帧独立 attention，每个 latent frame 只 attend 到自己对应的 action
      2) adaln_tmod: 将 action embedding 投影成 adaLN 调制项，与 time embedding 的 t_mod 相加
      3) add_residual: 将 action embedding 投影成 residual 并加到 hidden_states
    - 可学习 scale: 初始化为小正数，避免梯度死亡
    """

    def __init__(
        self,
        action_dim: int = 8,
        hidden_size: int = 3072,
        num_heads: int = 24,
        num_latent_frames: int = 20,
        inject_layers: list = None,
        num_layers: int = 30,
        qk_norm: bool = True,
        dropout: float = 0.0,
        initial_scale: float = 0.01,
        injection_type: str = "cross_attn_frame",
    ):
        super().__init__()

        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.num_latent_frames = num_latent_frames
        self.num_layers = num_layers

        if inject_layers is None:
            self.inject_layers = set(range(num_layers))
        else:
            self.inject_layers = set(inject_layers)

        self.injection_type = injection_type

        # Action embedding
        self.action_embed = nn.Sequential(
            nn.Linear(action_dim * 2, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        # Temporal position embedding
        self.temporal_pos_embed = nn.Parameter(
            torch.randn(1, num_latent_frames, hidden_size) * 0.02
        )

        # Cross-attention projections (cross_attn_frame)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # adaLN t_mod projection (adaln_tmod)
        self.action_to_tmod = nn.Linear(hidden_size, hidden_size * 6, bias=False)

        # residual projection (add_residual)
        self.action_to_residual = nn.Linear(hidden_size, hidden_size, bias=False)

        # QK Norm
        if qk_norm:
            self.q_norm = WanRMSNorm(self.head_dim, eps=1e-5)
            self.k_norm = WanRMSNorm(self.head_dim, eps=1e-5)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        # 可学习 scale，初始化为小正数
        self.output_scale = nn.Parameter(torch.ones(num_layers) * initial_scale)

        self.dropout = nn.Dropout(dropout)

        # build_action_emb / action_tmod / apply_action 直接使用显式参数，避免 checkpoint 的可变状态

    def should_inject(self, block_idx: int) -> bool:
        return block_idx in self.inject_layers

    def build_action_emb(
        self,
        action_binary: torch.Tensor,
        action_magnitude: torch.Tensor,
    ) -> torch.Tensor:
        # concat + MLP + temporal pos embed
        ...

    def action_tmod(self, action_emb: torch.Tensor) -> torch.Tensor:
        # (B, T, C) -> (B, T, 6, C)
        ...

    def apply_action(
        self,
        hidden_states: torch.Tensor,
        action_emb: torch.Tensor,
        t: int,
        h: int,
        w: int,
        block_idx: int,
    ) -> torch.Tensor:
        # cross_attn_frame / add_residual
        ...

    def forward(
        self,
        hidden_states: torch.Tensor,
        action_binary: torch.Tensor,
        action_magnitude: torch.Tensor,
        t: int, h: int, w: int,
        block_idx: int,
    ) -> torch.Tensor:
        action_emb = self.build_action_emb(action_binary, action_magnitude)
        return self.apply_action(hidden_states, action_emb, t, h, w, block_idx)


ACTION_CONTROLLER_CONFIGS = {
    "wan_ti2v_5b": {
        "action_dim": 8,
        "hidden_size": 3072,
        "num_heads": 24,
        "num_latent_frames": 20,
        "num_layers": 30,
        "inject_layers": None,
        "qk_norm": True,
        "initial_scale": 0.01,
        "injection_type": "cross_attn_frame",
    },
    "wan_ti2v_5b_sparse": {
        "action_dim": 8,
        "hidden_size": 3072,
        "num_heads": 24,
        "num_latent_frames": 20,
        "num_layers": 30,
        "inject_layers": list(range(0, 30, 3)),
        "qk_norm": True,
        "initial_scale": 0.01,
        "injection_type": "cross_attn_frame",
    },
    "wan_ti2v_5b_adaln": {
        "action_dim": 8,
        "hidden_size": 3072,
        "num_heads": 24,
        "num_latent_frames": 20,
        "num_layers": 30,
        "inject_layers": None,
        "qk_norm": True,
        "initial_scale": 0.01,
        "injection_type": "adaln_tmod",
    },
    "wan_ti2v_5b_add": {
        "action_dim": 8,
        "hidden_size": 3072,
        "num_heads": 24,
        "num_latent_frames": 20,
        "num_layers": 30,
        "inject_layers": None,
        "qk_norm": True,
        "initial_scale": 0.01,
        "injection_type": "add_residual",
    },
}
```

---

### 初始化说明

- Linear / projection 层使用 PyTorch 默认初始化（Kaiming uniform）
- `temporal_pos_embed` ~ N(0, 0.02)
- `output_scale` 使用 `initial_scale` 初始化（默认 0.01，可通过 `--action_initial_scale` 调整）

### 2. 修改 `wan_video.py`

#### 2.1 在 `WanVideoPipeline.__init__` 中添加

```python
class WanVideoPipeline(BasePipeline):
    def __init__(self, device="cuda", torch_dtype=torch.bfloat16):
        super().__init__(...)
        # ... existing code ...

        self.action_controller = None  # 新增

        # 更新 in_iteration_models
        self.in_iteration_models = ("dit", "motion_controller", "vace", "animate_adapter", "vap", "action_controller")
        self.in_iteration_models_2 = ("dit2", "motion_controller", "vace2", "animate_adapter", "vap", "action_controller")
```

#### 2.2 修改 `model_fn_wan_video`

```python
def model_fn_wan_video(
    dit: WanModel,
    motion_controller: WanMotionControllerModel = None,
    vace: VaceWanModel = None,
    vap: MotWanModel = None,
    animate_adapter: WanAnimateAdapter = None,
    action_controller = None,  # 新增
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    clip_feature: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    # ... existing params ...
    action_binary: Optional[torch.Tensor] = None,      # 新增
    action_magnitude: Optional[torch.Tensor] = None,   # 新增
    **kwargs,
):
    # ... existing code until block loop ...

    # 预计算 action embedding
    action_emb = None
    if action_controller is not None and action_binary is not None:
        if not torch.is_tensor(action_binary):
            action_binary = torch.tensor(action_binary)
        if action_magnitude is None:
            action_magnitude = torch.zeros_like(action_binary)
        elif not torch.is_tensor(action_magnitude):
            action_magnitude = torch.tensor(action_magnitude)
        if action_binary.dim() == 2:
            action_binary = action_binary.unsqueeze(0)
        if action_magnitude.dim() == 2:
            action_magnitude = action_magnitude.unsqueeze(0)
        action_binary = action_binary.to(device=x.device, dtype=x.dtype)
        action_magnitude = action_magnitude.to(device=x.device, dtype=x.dtype)
        action_emb = action_controller.build_action_emb(action_binary, action_magnitude)

        # adaln_tmod: 将 action embedding 投影为 t_mod 的增量并融合
        if action_controller.injection_type == "adaln_tmod":
            action_tmod = action_controller.action_tmod(action_emb)
            # NOTE: 这里省略对齐细节，实际实现会按帧/Token 对齐后再相加
            t_mod = t_mod + action_tmod

    for block_id, block in enumerate(dit.blocks):
        # Block forward (existing)
        if use_gradient_checkpointing:
            x = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                x, context, t_mod, freqs,
                use_reentrant=False,
            )
        else:
            x = block(x, context, t_mod, freqs)

        # Action Controller 注入
        if action_controller is not None and action_emb is not None and action_controller.injection_type != "adaln_tmod":
            if action_controller.should_inject(block_id):
                if use_gradient_checkpointing:
                    def create_action_forward(controller, t, h, w, block_idx):
                        def custom_forward(hidden_states, action_emb):
                            return controller.apply_action(hidden_states, action_emb, t, h, w, block_idx)
                        return custom_forward
                    x = torch.utils.checkpoint.checkpoint(
                        create_action_forward(action_controller, f, h, w, block_id),
                        x, action_emb,
                        use_reentrant=False,
                    )
                else:
                    x = action_controller.apply_action(x, action_emb, f, h, w, block_id)

        # VACE, Animate 等 existing code...

    # ... rest of the function ...
```

---

### 3. `train_action.py`

```python
# examples/wanvideo/model_training/train_action.py

import torch, os, argparse, accelerate, warnings, json
from safetensors.torch import load_file as load_safetensors
from diffsynth.core import UnifiedDataset
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.models.wan_video_action_controller import WanActionController, ACTION_CONTROLLER_CONFIGS
from diffsynth.diffusion import *
os.environ["TOKENIZERS_PARALLELISM"] = "false"


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

        actual_device = device

        # Load base models
        model_configs = self.parse_model_configs(
            model_paths, model_id_with_origin_paths,
            fp8_models=fp8_models, offload_models=offload_models, device=device,
            model_dir=model_dir
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

        # 创建 Action Controller
        if isinstance(action_controller_config, str):
            ac_config = ACTION_CONTROLLER_CONFIGS[action_controller_config].copy()
        else:
            ac_config = action_controller_config.copy()
        if action_injection_type is not None:
            ac_config["injection_type"] = action_injection_type
        if action_initial_scale is not None:
            ac_config["initial_scale"] = action_initial_scale

        self.pipe.action_controller = WanActionController(**ac_config)
        self.pipe.action_controller = self.pipe.action_controller.to(
            device=actual_device,
            dtype=torch.bfloat16
        )

        if action_controller_checkpoint is not None:
            if action_controller_checkpoint.endswith('.safetensors'):
                state_dict = load_safetensors(action_controller_checkpoint)
            else:
                state_dict = torch.load(action_controller_checkpoint, map_location="cpu")
            self.pipe.action_controller.load_state_dict(state_dict)
            print(f"Loaded action controller checkpoint: {action_controller_checkpoint}")

        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint,
            task=task,
        )

        if trainable_models is not None and "action_controller" in trainable_models.split(","):
            self.pipe.action_controller.train()
            self.pipe.action_controller.requires_grad_(True)
            print(f"Action controller trainable params: {sum(p.numel() for p in self.pipe.action_controller.parameters() if p.requires_grad)}")
        else:
            self.pipe.action_controller.eval()
            self.pipe.action_controller.requires_grad_(False)

        # Magnitude 归一化系数（Forward/Back/Left/Right 乘 100）
        self.action_magnitude_scale = torch.tensor([100, 100, 100, 100, 1, 1, 1, 1], dtype=torch.float32)

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
            return None, None

        if isinstance(action_str_or_tensor, str):
            action = json.loads(action_str_or_tensor)
            action = torch.tensor(action, dtype=torch.float32)
        elif isinstance(action_str_or_tensor, list):
            action = torch.tensor(action_str_or_tensor, dtype=torch.float32)
        else:
            action = action_str_or_tensor

        if len(action.shape) == 3:
            action = action.unsqueeze(0)

        action_binary = action[:, :, 0, :]
        action_magnitude = action[:, :, 1, :]

        # 统一动作量级（训练/推理需一致）
        action_magnitude = action_magnitude * self.action_magnitude_scale.to(action_magnitude.device, action_magnitude.dtype)

        return action_binary, action_magnitude

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

        # 解析 action
        if "action" in data:
            action_binary, action_magnitude = self.parse_action(data["action"])
            if action_binary is not None:
                inputs_shared["action_binary"] = action_binary
                inputs_shared["action_magnitude"] = action_magnitude

        # 首帧作为 input_image (用于生成 first_frame_latents)
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

        if remove_prefix is not None:
            state_dict_ = {}
            for name, param in state_dict.items():
                new_name = name
                if name.startswith("pipe.action_controller."):
                    new_name = name[len("pipe.action_controller."):]
                elif remove_prefix and name.startswith(remove_prefix):
                    new_name = name[len(remove_prefix):]
                state_dict_[new_name] = param
            state_dict = state_dict_
        return state_dict


def action_parser():
    parser = argparse.ArgumentParser(description="Wan Video + Action Controller Training")
    parser = add_general_config(parser)
    parser = add_video_size_config(parser)
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--action_controller_config", type=str, default="wan_ti2v_5b",
                        choices=list(ACTION_CONTROLLER_CONFIGS.keys()))
    parser.add_argument("--action_injection_type", type=str, default=None,
                        choices=["cross_attn_frame", "adaln_tmod", "add_residual"])
    parser.add_argument("--action_initial_scale", type=float, default=None)
    parser.add_argument("--action_controller_checkpoint", type=str, default=None)
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0)
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0)
    parser.add_argument("--initialize_model_on_cpu", default=False, action="store_true")
    return parser


if __name__ == "__main__":
    parser = action_parser()
    args = parser.parse_args()

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

    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt="pipe.action_controller.",
    )

    # 保存训练配置，供推理与复现实验使用
    action_cfg = ACTION_CONTROLLER_CONFIGS[args.action_controller_config].copy()
    if args.action_injection_type is not None:
        action_cfg["injection_type"] = args.action_injection_type
    if args.action_initial_scale is not None:
        action_cfg["initial_scale"] = args.action_initial_scale
    save_config = {
        "base_model_id": "Wan-AI/Wan2.2-TI2V-5B",
        "action_controller_config": action_cfg,
        "action_magnitude_scale": [100, 100, 100, 100, 1, 1, 1, 1],
        "lora": {
            "lora_base_model": args.lora_base_model,
            "lora_target_modules": args.lora_target_modules,
            "lora_rank": args.lora_rank,
        },
    }
    os.makedirs(args.output_path, exist_ok=True)
    with open(os.path.join(args.output_path, "action_train_config.json"), "w") as f:
        json.dump(save_config, f, indent=2)

    # 额外保存 action controller 与 LoRA 权重（用于推理加载）
    if accelerator.is_main_process:
        ac_state = model.pipe.action_controller.state_dict()
        accelerator.save(ac_state, os.path.join(args.output_path, "action_controller.safetensors"), safe_serialization=True)
        full_state = accelerator.get_state_dict(model)
        lora_state = {k.replace("pipe.dit.", ""): v for k, v in full_state.items() if ".lora_" in k}
        if lora_state:
            accelerator.save(lora_state, os.path.join(args.output_path, "lora.safetensors"), safe_serialization=True)

    launch_training_task(accelerator, dataset, model, model_logger, args=args)
```

---

### 4. `train_action.sh`

```bash
#!/bin/bash

# DiffSynth-Studio/train_action.sh
export DIFFSYNTH_ATTENTION_IMPLEMENTATION=torch

DATASET_BASE=/fs/cml-projects/worldmodel/Self-Forcing/dataset_download/Sekai-Project/sekai-game-drone_v2_processed
METADATA_PATH=${DATASET_BASE}/clips_metadata_diffsynth.csv

# 注意: CSV 列名需要是 "video" 而非 "video_name"

accelerate launch examples/wanvideo/model_training/train_action.py \
    --dataset_base_path "${DATASET_BASE}" \
    --dataset_metadata_path "${METADATA_PATH}" \
    --data_file_keys "video" \
    --dataset_num_workers 16 \
    --model_dir "/path/to/Wan2.2-TI2V-5B-local" \
    --trainable_models "action_controller" \
    --action_controller_config "wan_ti2v_5b" \
    --action_injection_type "cross_attn_frame" \
    --action_initial_scale 0.01 \
    --learning_rate 1e-4 \
    --lr_warmup_ratio 0.05 \
    --lr_min_lr 5e-6 \
    --weight_decay 0.01 \
    --max_steps 40000 \
    --log_steps 10 \
    --save_steps 5000 \
    --use_wandb \
    --wandb_project "wan-ti2v-action" \
    --wandb_run_name "action-controller" \
    --gradient_accumulation_steps 4 \
    --height 480 \
    --width 832 \
    --num_frames 81 \
    --output_path "./models/train/Wan2.2-TI2V-5B_action_controller"
```

如需兼容旧方式，可将 `--model_dir` 替换为：
`--model_id_with_origin_paths "Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors, Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth, Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth"`

---

## 首帧处理 (TI2V 首帧融合)

Wan2.2-TI2V-5B 自带 `fuse_vae_embedding_in_latents=True`，首帧会通过 `WanVideoUnit_ImageEmbedderFused` 融合到 latents。  
因此无需新增 “clean latent” 单元或在 denoising 后强行 clamp。训练/推理只需提供 `input_image` 即可保持一致。

---

## 保存与加载（Action Module + 可选 LoRA + Config）

为便于 ablation 与复现，训练阶段建议保存三类文件：

```
output_action_controller/
├── action_train_config.json          # 记录 injection_type、initial_scale、base_model_id/base_model_dir
├── action_controller.safetensors     # 仅 action controller 权重
├── lora.safetensors                  # 可选：仅 LoRA 权重（从训练模型中筛出）
└── checkpoints/
    └── step-000010000/
        ├── trainable.safetensors         # 训练态权重快照（按训练脚本的 remove_prefix 保存）
        ├── trainable_full.safetensors    # Resume 用（包含完整参数名，便于直接 load_state_dict）
        ├── trainer_state.json            # global_step/epoch
        ├── optimizer.pt                  # optimizer state（轻量）
        └── scheduler.pt                  # scheduler state（轻量，可选）
```

**保存策略（训练中或训练结束后执行）：**
- `action_controller.safetensors`: `pipe.action_controller.state_dict()` 直接保存。
- `lora.safetensors`: 如使用 LoRA，再从训练模型 `state_dict` 中筛选 `".lora_"` 相关权重，并去掉 `pipe.dit.` 前缀，必要时使用 `mapping_lora_state_dict` 做键名兼容。
- `action_train_config.json`: 已在训练脚本中保存，包含 `action_controller_config`（含 `injection_type`/`initial_scale`）与 LoRA 配置（若使用）。
- `checkpoints/step-*/`: 需显式开启 `--save_full_state` 才会保存，可用于 `--resume_from` 恢复训练。

**Resume 训练：**
- `--save_full_state` 写入轻量训练态（trainable + optimizer/scheduler），不再保存完整模型权重与 RNG。
- `--resume_from /path/to/checkpoints/step-000010000` 显式恢复。
- `--max_checkpoints N` 控制保留最近 N 份 checkpoint。

**推理加载策略：**
1) 读取 `action_train_config.json` 获取 `action_controller_config` 和 LoRA 配置（若有）。  
2) 构造 `WanActionController`（`injection_type`/`initial_scale` 由 config 决定）并加载 `action_controller.safetensors`。  
3) 如配置了 LoRA，再对 `pipe.dit` 注入 LoRA（`lora_base_model`, `lora_target_modules`, `lora_rank`），并加载 `lora.safetensors`。  

---

## LoRA Merge（用于下一阶段训练）

```bash
python scripts/merge_lora_to_base.py \
  --model_dir "/path/to/Wan2.2-TI2V-5B-local" \
  --lora_path "/path/to/lora.safetensors" \
  --alpha 1.0 \
  --output_dir "./models/merged/Wan2.2-TI2V-5B-merged"
```

默认输出目录为 `./models/merged/<base_name>_merged`，如需覆盖请加 `--overwrite`。

---

## 推理代码

```python
import json
import torch
from PIL import Image
from safetensors.torch import load_file as load_safetensors
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.models.wan_video_action_controller import WanActionController
from diffsynth.diffusion.training_module import DiffusionTrainingModule

# Load pipeline (TI2V 5B)
pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="Wan2.2_VAE.pth"),
    ]
)

# Load training config
with open("./output_action_controller/action_train_config.json", "r") as f:
    cfg = json.load(f)

# Load action controller
ac_config = cfg["action_controller_config"]
pipe.action_controller = WanActionController(**ac_config)
ac_state = load_safetensors("./output_action_controller/action_controller.safetensors")
pipe.action_controller.load_state_dict(ac_state)
pipe.action_controller = pipe.action_controller.to(pipe.device, pipe.torch_dtype)
pipe.action_controller.eval()

# Load LoRA (if configured)
lora_cfg = cfg.get("lora", {})
if lora_cfg.get("lora_base_model"):
    helper = DiffusionTrainingModule()
    pipe.dit = helper.add_lora_to_model(
        pipe.dit,
        target_modules=lora_cfg["lora_target_modules"].split(","),
        lora_rank=lora_cfg["lora_rank"],
        upcast_dtype=pipe.torch_dtype,
    )
    lora_state = load_safetensors("./output_action_controller/lora.safetensors")
    lora_state = helper.mapping_lora_state_dict(lora_state)
    pipe.dit.load_state_dict(lora_state, strict=False)

# Inputs
first_frame = Image.open("first_frame.jpg")
prompt = "A drone flying forward over mountains"

# Action: moving forward
action_binary = torch.zeros(1, 20, 8)
action_binary[:, :, 0] = 1  # Forward
action_magnitude = torch.zeros(1, 20, 8)
action_magnitude[:, :, 0] = 0.5
action_magnitude = action_magnitude * torch.tensor([100, 100, 100, 100, 1, 1, 1, 1])

# Generate
video = pipe(
    prompt=prompt,
    input_image=first_frame,  # 首帧 clean latent
    action_binary=action_binary.to(pipe.device, pipe.torch_dtype),
    action_magnitude=action_magnitude.to(pipe.device, pipe.torch_dtype),
    height=480,
    width=832,
    num_frames=81,
    num_inference_steps=50,
)

# Save
for i, frame in enumerate(video):
    frame.save(f"output/frame_{i:04d}.png")
```

---

## 实现注意事项

1. **量级统一**: Forward/Back/Left/Right 的 magnitude 必须乘 100（训练与推理保持一致）。建议把 scale 写进 `action_train_config.json` 并在推理时读取。
2. **CFG 负分支**: 如果 `cfg_merge=False`，action 放在 `inputs_shared` 会自动对正/负分支生效；若 `cfg_merge=True`，会对 action 做 concat。列表输入会在 pipeline 里转成 tensor，但仍建议传 `(B, T, 8)`。
3. **时间对齐**: `action_len = latent_frames - 1`；若 `num_frames` 或时间压缩比改变，action 长度必须同步调整。
4. **dtype/device**: `action_binary/action_magnitude` 需和 `pipe.torch_dtype/device` 对齐，避免不必要的类型转换。
5. **adaln_tmod**: `action_tmod` 形状需与 `t_mod` 对齐（TI2V 的 `seperated_timestep` 情况尤其注意）。
6. **初始化**: 默认 `initial_scale=0.01` 让 action 初期影响很小且可学习；若要严格零影响，可设置 `--action_initial_scale 0.0`（action 参数会在 scale 变大后才开始更新）。
7. **注入稀疏层**: `inject_layers` 改动需要写入 config，确保训练/推理一致。

---

## 关键设计点总结

1. **模型**: Wan2.2-TI2V-5B (hidden=3072, heads=24, layers=30)
2. **首帧**: 使用 TI2V 的 `fuse_vae_embedding_in_latents` 机制
3. **Action**: 20个，对应 latent frame 1-20
4. **注入方式**: 逐帧独立 cross-attention
5. **注入超参**: `injection_type` ∈ {`cross_attn_frame`, `adaln_tmod`, `add_residual`}，支持全层/稀疏注入
6. **初始化**: `initial_scale=0.01` 默认让 action 初期影响很小；如需严格零影响，可设为 0.0
7. **Checkpoint**: 用闭包捕获 `t/h/w/block_idx`，checkpoint 内调用 `apply_action`
8. **保存**: `action_controller.safetensors` + `action_train_config.json`（`lora.safetensors` 可选）
