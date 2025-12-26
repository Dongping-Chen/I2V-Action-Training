import json
import os
import shutil
import torch
from accelerate import Accelerator


class ModelLogger:
    def __init__(
        self,
        output_path,
        remove_prefix_in_ckpt=None,
        state_dict_converter=lambda x:x,
        save_full_state=False,
        max_checkpoints=None,
        checkpoint_metadata=None,
        extra_checkpoint_saver=None,
        save_root_steps=True,
        checkpoint_weights_name="trainable.safetensors",
    ):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter
        self.save_full_state = save_full_state
        self.max_checkpoints = max_checkpoints
        self.checkpoint_metadata = checkpoint_metadata
        self.extra_checkpoint_saver = extra_checkpoint_saver
        self.save_root_steps = save_root_steps
        self.checkpoint_weights_name = checkpoint_weights_name
        self.checkpoints_path = os.path.join(output_path, "checkpoints")
        self.num_steps = 0
        self.last_checkpoint_step = None


    def set_num_steps(self, num_steps):
        self.num_steps = num_steps


    def on_step_end(
        self,
        accelerator: Accelerator,
        model: torch.nn.Module,
        save_steps=None,
        epoch_id=None,
        optimizer=None,
        scheduler=None,
    ):
        self.num_steps += 1
        if save_steps is not None and self.num_steps % save_steps == 0:
            if self.save_root_steps:
                self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")
            if self.save_full_state:
                self.save_checkpoint(
                    accelerator,
                    model,
                    self.num_steps,
                    epoch_id=epoch_id,
                    optimizer=optimizer,
                    scheduler=scheduler,
                )


    def on_epoch_end(
        self,
        accelerator: Accelerator,
        model: torch.nn.Module,
        epoch_id,
        optimizer=None,
        scheduler=None,
    ):
        if self.save_root_steps:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                state_dict = self._export_trainable_state_dict(
                    accelerator,
                    model,
                    remove_prefix=self.remove_prefix_in_ckpt,
                )
                os.makedirs(self.output_path, exist_ok=True)
                path = os.path.join(self.output_path, f"epoch-{epoch_id}.safetensors")
                accelerator.save(state_dict, path, safe_serialization=True)
        if self.save_full_state:
            self.save_checkpoint(
                accelerator,
                model,
                self.num_steps,
                epoch_id=epoch_id,
                optimizer=optimizer,
                scheduler=scheduler,
            )


    def on_training_end(
        self,
        accelerator: Accelerator,
        model: torch.nn.Module,
        save_steps=None,
        epoch_id=None,
        optimizer=None,
        scheduler=None,
    ):
        if save_steps is not None and self.num_steps % save_steps != 0:
            if self.save_root_steps:
                self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")
            if self.save_full_state:
                self.save_checkpoint(
                    accelerator,
                    model,
                    self.num_steps,
                    epoch_id=epoch_id,
                    optimizer=optimizer,
                    scheduler=scheduler,
                )
        elif self.save_full_state and self.last_checkpoint_step != self.num_steps:
            self.save_checkpoint(
                accelerator,
                model,
                self.num_steps,
                epoch_id=epoch_id,
                optimizer=optimizer,
                scheduler=scheduler,
            )


    def save_model(self, accelerator: Accelerator, model: torch.nn.Module, file_name):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, file_name)
            self.save_model_to(accelerator, model, path)


    def _export_trainable_state_dict(self, accelerator: Accelerator, model: torch.nn.Module, remove_prefix=None):
        state_dict = accelerator.get_state_dict(model)
        state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=remove_prefix)
        state_dict = self.state_dict_converter(state_dict)
        return state_dict


    def _nested_to_cpu(self, obj):
        if torch.is_tensor(obj):
            return obj.detach().cpu()
        if isinstance(obj, dict):
            return {k: self._nested_to_cpu(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            converted = [self._nested_to_cpu(v) for v in obj]
            return type(obj)(converted)
        return obj


    def save_model_to(self, accelerator: Accelerator, model: torch.nn.Module, path: str):
        state_dict = self._export_trainable_state_dict(
            accelerator,
            model,
            remove_prefix=self.remove_prefix_in_ckpt,
        )
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        accelerator.save(state_dict, path, safe_serialization=True)


    def save_checkpoint(
        self,
        accelerator: Accelerator,
        model: torch.nn.Module,
        step: int,
        epoch_id=None,
        optimizer=None,
        scheduler=None,
    ):
        if step is None:
            return
        if self.last_checkpoint_step == step:
            return
        checkpoint_dir = os.path.join(self.checkpoints_path, f"step-{step:08d}")
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            os.makedirs(checkpoint_dir, exist_ok=True)
            trainable_path = os.path.join(checkpoint_dir, self.checkpoint_weights_name)
            self.save_model_to(accelerator, model, trainable_path)
            trainer_state = {
                "global_step": step,
                "epoch": epoch_id,
                "weights_name": self.checkpoint_weights_name,
                "remove_prefix_in_ckpt": self.remove_prefix_in_ckpt,
            }
            with open(os.path.join(checkpoint_dir, "trainer_state.json"), "w", encoding="utf-8") as f:
                json.dump(trainer_state, f, indent=2)
            if self.checkpoint_metadata is not None:
                config_path = os.path.join(checkpoint_dir, "train_config.json")
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(self.checkpoint_metadata, f, indent=2, ensure_ascii=True, default=str)
            if self.extra_checkpoint_saver is not None:
                self.extra_checkpoint_saver(checkpoint_dir, accelerator, model)
            if optimizer is not None:
                optimizer_state = self._nested_to_cpu(optimizer.state_dict())
                torch.save(optimizer_state, os.path.join(checkpoint_dir, "optimizer.pt"))
            if scheduler is not None:
                scheduler_state = self._nested_to_cpu(scheduler.state_dict())
                torch.save(scheduler_state, os.path.join(checkpoint_dir, "scheduler.pt"))
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            self.prune_checkpoints()
        self.last_checkpoint_step = step


    def prune_checkpoints(self):
        if self.max_checkpoints is None:
            return
        if not os.path.isdir(self.checkpoints_path):
            return
        entries = []
        for name in os.listdir(self.checkpoints_path):
            if not name.startswith("step-"):
                continue
            step_str = name.replace("step-", "")
            if not step_str.isdigit():
                continue
            entries.append((int(step_str), name))
        entries.sort(key=lambda x: x[0])
        if len(entries) <= self.max_checkpoints:
            return
        for _, name in entries[:-self.max_checkpoints]:
            shutil.rmtree(os.path.join(self.checkpoints_path, name), ignore_errors=True)
