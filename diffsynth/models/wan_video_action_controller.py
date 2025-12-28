import math
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange

from .wan_video_dit import CrossAttention


class WanActionController(nn.Module):
    def __init__(
        self,
        action_dim: int = 8,
        action_text_dim: int = 4096,
        hidden_size: int = 3072,
        num_heads: int = 24,
        num_latent_frames: int = 20,
        inject_layers: Optional[list] = None,
        inject_every_n: Optional[int] = None,
        num_layers: int = 30,
        dropout: float = 0.0,
        initial_scale: float = 0.0,
        injection_type: str = "patch_add",
        **kwargs,
    ):
        super().__init__()

        self.action_dim = action_dim
        self.action_text_dim = action_text_dim
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_latent_frames = num_latent_frames
        self.num_layers = num_layers
        self.injection_type = injection_type
        if injection_type not in ("patch_add", "layer_add", "cross_attn"):
            raise ValueError(f"Unknown injection_type: {injection_type}")
        if kwargs:
            _ = kwargs

        if inject_layers is None:
            if inject_every_n is None:
                self.inject_layers = set(range(num_layers))
            else:
                self.inject_layers = set(range(0, num_layers, inject_every_n))
        else:
            self.inject_layers = set(inject_layers)

        self.action_embed = nn.Sequential(
            nn.Linear(action_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.action_text_embed = nn.Sequential(
            nn.Linear(action_text_dim + action_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.action_to_residual = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size, bias=False) for _ in range(num_layers)]
        )

        initial_scale = float(initial_scale)
        self.register_buffer("action_gate", torch.ones(num_layers) * initial_scale)
        self.register_buffer("action_patch_gate", torch.tensor(initial_scale))
        self.dropout = nn.Dropout(dropout)

        self.action_cross_attn = None
        self._initialized_from_dit = False
        if self.injection_type == "cross_attn":
            self.action_cross_attn = nn.ModuleList(
                [
                    CrossAttention(
                        dim=hidden_size,
                        num_heads=num_heads,
                        has_image_input=False,
                    )
                    for _ in range(num_layers)
                ]
            )

    def _sinusoidal_temporal_pos_embed(self, length: int, device, dtype) -> torch.Tensor:
        half_dim = self.hidden_size // 2
        if half_dim == 0:
            return torch.zeros(1, length, self.hidden_size, device=device, dtype=dtype)
        positions = torch.arange(length, device=device, dtype=torch.float32)
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half_dim, device=device, dtype=torch.float32)
            / half_dim
        )
        angles = positions[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if self.hidden_size % 2 == 1:
            emb = torch.cat([emb, torch.zeros(length, 1, device=device)], dim=-1)
        return emb.unsqueeze(0).to(dtype=dtype)

    def _get_temporal_pos_embed(self, length: int, device, dtype) -> torch.Tensor:
        return self._sinusoidal_temporal_pos_embed(length, device, dtype)

    def should_inject(self, block_idx: int) -> bool:
        return block_idx in self.inject_layers

    def _align_action_emb(self, action_emb: torch.Tensor, num_frames_need_action: int) -> torch.Tensor:
        if action_emb.shape[1] < num_frames_need_action:
            pad_action = action_emb[:, -1:, :].expand(-1, num_frames_need_action - action_emb.shape[1], -1)
            action_emb = torch.cat([action_emb, pad_action], dim=1)
        elif action_emb.shape[1] > num_frames_need_action:
            action_emb = action_emb[:, :num_frames_need_action, :]
        return action_emb

    def _align_action_context(
        self,
        action_context: torch.Tensor,
        action_context_mask: Optional[torch.Tensor],
        num_frames_need_action: int,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if action_context.shape[1] < num_frames_need_action:
            pad_ctx = action_context[:, -1:, :, :].expand(-1, num_frames_need_action - action_context.shape[1], -1, -1)
            action_context = torch.cat([action_context, pad_ctx], dim=1)
            if action_context_mask is not None:
                pad_mask = action_context_mask[:, -1:, :].expand(-1, num_frames_need_action - action_context_mask.shape[1], -1)
                action_context_mask = torch.cat([action_context_mask, pad_mask], dim=1)
        elif action_context.shape[1] > num_frames_need_action:
            action_context = action_context[:, :num_frames_need_action, :, :]
            if action_context_mask is not None:
                action_context_mask = action_context_mask[:, :num_frames_need_action, :]
        if action_context_mask is not None and action_context_mask.numel() > 0:
            action_context_mask = action_context_mask.clone()
            action_context_mask[..., 0] = True
        return action_context, action_context_mask

    def init_from_dit(self, dit) -> None:
        if self._initialized_from_dit:
            return
        if self.action_cross_attn is None:
            self._initialized_from_dit = True
            return
        if dit is None:
            raise ValueError("init_from_dit() failed: dit is None.")
        if not hasattr(dit, "blocks"):
            raise ValueError("init_from_dit() failed: dit has no attribute 'blocks'.")
        blocks = list(getattr(dit, "blocks", []))
        if len(blocks) < len(self.action_cross_attn):
            raise ValueError(
                "init_from_dit() failed: dit.blocks is shorter than action_cross_attn "
                f"({len(blocks)} < {len(self.action_cross_attn)})."
            )
        for idx in range(len(self.action_cross_attn)):
            block = blocks[idx]
            if not hasattr(block, "cross_attn"):
                raise ValueError(f"init_from_dit() failed: dit.blocks[{idx}] has no 'cross_attn'.")
            load_result = self.action_cross_attn[idx].load_state_dict(block.cross_attn.state_dict(), strict=False)
            if getattr(load_result, "missing_keys", None):
                raise ValueError(
                    f"init_from_dit() failed: missing keys when copying cross_attn for layer {idx}: "
                    f"{load_result.missing_keys}"
                )
        self._initialized_from_dit = True

    def build_action_emb(self, action_magnitude: torch.Tensor) -> torch.Tensor:
        action_emb = self.action_embed(action_magnitude)
        pos_emb = self._get_temporal_pos_embed(
            action_emb.shape[1],
            device=action_emb.device,
            dtype=action_emb.dtype,
        )
        if pos_emb.shape[1] != action_emb.shape[1]:
            action_emb = action_emb[:, :pos_emb.shape[1], :]
        action_emb = action_emb + pos_emb
        return action_emb

    def build_action_emb_from_text(
        self,
        action_text_embeds: torch.Tensor,
        action_magnitude: torch.Tensor,
    ) -> torch.Tensor:
        if action_text_embeds is None or action_magnitude is None:
            raise ValueError("action_text_embeds and action_magnitude are required for text-based action embedding.")
        if action_text_embeds.dim() == 2:
            action_text_embeds = action_text_embeds.unsqueeze(0)
        if action_magnitude.dim() == 2:
            action_magnitude = action_magnitude.unsqueeze(0)
        if action_text_embeds.shape[1] != action_magnitude.shape[1]:
            action_text_embeds = self._align_action_emb(action_text_embeds, action_magnitude.shape[1])
            action_magnitude = action_magnitude[:, :action_text_embeds.shape[1], :]
        text_and_mag = torch.cat([action_text_embeds, action_magnitude], dim=-1)
        action_emb = self.action_text_embed(text_and_mag)
        pos_emb = self._get_temporal_pos_embed(
            action_emb.shape[1],
            device=action_emb.device,
            dtype=action_emb.dtype,
        )
        if pos_emb.shape[1] != action_emb.shape[1]:
            action_emb = action_emb[:, :pos_emb.shape[1], :]
        action_emb = action_emb + pos_emb
        return action_emb

    def apply_action(
        self,
        hidden_states: torch.Tensor,
        action_emb: torch.Tensor,
        t: int,
        h: int,
        w: int,
        block_idx: int,
    ) -> torch.Tensor:
        if not self.should_inject(block_idx):
            return hidden_states
        if self.injection_type != "layer_add":
            return hidden_states
        num_frames_need_action = t - 1
        action_emb = self._align_action_emb(action_emb, num_frames_need_action)
        residual = self.action_to_residual[block_idx](action_emb)
        zeros_first = torch.zeros(residual.shape[0], 1, residual.shape[2], device=residual.device, dtype=residual.dtype)
        residual_full = torch.cat([zeros_first, residual], dim=1)
        residual_full = residual_full.repeat_interleave(h * w, dim=1)
        gate = self.action_gate[block_idx].to(device=hidden_states.device, dtype=hidden_states.dtype)
        return hidden_states + gate * residual_full

    def apply_action_cross_attn(
        self,
        hidden_states: torch.Tensor,
        hidden_states_norm: torch.Tensor,
        action_context: Optional[torch.Tensor],
        action_context_mask: Optional[torch.Tensor],
        t: int,
        h: int,
        w: int,
        block_idx: int,
    ) -> torch.Tensor:
        if not self.should_inject(block_idx):
            return hidden_states
        if self.injection_type != "cross_attn":
            return hidden_states
        if self.action_cross_attn is None or action_context is None:
            return hidden_states
        if not self._initialized_from_dit:
            raise RuntimeError("WanActionController(cross_attn) must call init_from_dit(dit) before use.")
        if t is None or h is None or w is None:
            return hidden_states
        tokens_per_frame = h * w
        if tokens_per_frame <= 0:
            return hidden_states
        num_frames_need_action = max(t - 1, 0)
        if num_frames_need_action == 0:
            return hidden_states

        if action_context.dim() == 3:
            action_context = action_context.unsqueeze(2)
        if action_context_mask is not None and action_context_mask.dim() == 2:
            action_context_mask = action_context_mask.unsqueeze(2)

        action_context, action_context_mask = self._align_action_context(
            action_context,
            action_context_mask,
            num_frames_need_action,
        )

        x = rearrange(hidden_states, "b (t s) c -> b t s c", t=t, s=tokens_per_frame)
        x_norm = rearrange(hidden_states_norm, "b (t s) c -> b t s c", t=t, s=tokens_per_frame)
        x_act = x[:, 1:]
        x_norm_act = x_norm[:, 1:]

        bsz = x.shape[0]
        query = rearrange(x_norm_act, "b t s c -> (b t) s c")
        ctx = rearrange(action_context, "b t k c -> (b t) k c")
        if action_context_mask is not None:
            ctx_mask = rearrange(action_context_mask, "b t k -> (b t) k")
        else:
            ctx_mask = None

        residual = self.action_cross_attn[block_idx](query, ctx, y_mask=ctx_mask)
        residual = self.dropout(residual)
        residual = rearrange(residual, "(b t) s c -> b t s c", b=bsz, t=num_frames_need_action)
        gate = self.action_gate[block_idx].to(device=hidden_states.device, dtype=hidden_states.dtype)
        x_act = x_act + gate * residual
        x = torch.cat([x[:, :1], x_act], dim=1)
        return rearrange(x, "b t s c -> b (t s) c")

    def apply_patch_add(
        self,
        hidden_states: torch.Tensor,
        action_emb: torch.Tensor,
        t: int,
        h: int,
        w: int,
    ) -> torch.Tensor:
        if action_emb is None:
            return hidden_states
        if t is None or h is None or w is None:
            return hidden_states
        tokens_per_frame = h * w
        if tokens_per_frame <= 0:
            return hidden_states
        num_frames_need_action = max(t - 1, 0)
        if num_frames_need_action == 0:
            return hidden_states
        action_emb = self._align_action_emb(action_emb, num_frames_need_action)
        B = hidden_states.shape[0]
        gate = self.action_patch_gate.to(device=hidden_states.device, dtype=hidden_states.dtype)
        zeros_first = torch.zeros(B, 1, action_emb.shape[2], device=hidden_states.device, dtype=hidden_states.dtype)
        action_full = torch.cat([zeros_first, action_emb], dim=1)
        action_full = action_full[:, :, None, :].expand(-1, -1, tokens_per_frame, -1)
        hidden_reshaped = rearrange(hidden_states, "b (t s) c -> b t s c", t=t, s=tokens_per_frame)
        hidden_reshaped = hidden_reshaped + gate * action_full
        return rearrange(hidden_reshaped, "b t s c -> b (t s) c")

    def forward(
        self,
        hidden_states: torch.Tensor,
        action_magnitude: torch.Tensor,
        t: int,
        h: int,
        w: int,
        block_idx: int,
    ) -> torch.Tensor:
        action_emb = self.build_action_emb(action_magnitude)
        return self.apply_action(hidden_states, action_emb, t, h, w, block_idx)


ACTION_CONTROLLER_CONFIGS = {
    "wan_ti2v_5b": {
        "action_dim": 8,
        "hidden_size": 3072,
        "num_latent_frames": 20,
        "num_layers": 30,
        "inject_layers": None,
        "inject_every_n": None,
        "initial_scale": 0.0,
        "injection_type": "patch_add",
    },
    "wan_ti2v_5b_layer": {
        "action_dim": 8,
        "hidden_size": 3072,
        "num_latent_frames": 20,
        "num_layers": 30,
        "inject_layers": None,
        "inject_every_n": None,
        "initial_scale": 0.0,
        "injection_type": "layer_add",
    },
    "wan_ti2v_5b_cross_attn": {
        "action_dim": 8,
        "hidden_size": 3072,
        "num_heads": 24,
        "num_latent_frames": 20,
        "num_layers": 30,
        "inject_layers": None,
        "inject_every_n": None,
        "initial_scale": 0.0,
        "injection_type": "cross_attn",
    },
}
