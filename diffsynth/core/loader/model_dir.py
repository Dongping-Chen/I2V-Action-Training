import glob
import json
import os


def load_model_index(model_dir: str) -> dict | None:
    for name in ["diffsynth_model_index.json", "model_index.json"]:
        path = os.path.join(model_dir, name)
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    return None


def _first_existing_pattern(model_dir: str, patterns: list[str]) -> str | None:
    for p in patterns:
        if glob.glob(os.path.join(model_dir, p)):
            return p
    return None


def _infer_components(model_dir: str) -> dict:
    components = {}

    dit = _first_existing_pattern(model_dir, [
        "dit/diffusion_pytorch_model*.safetensors",
        "transformer/diffusion_pytorch_model*.safetensors",
        "diffusion_pytorch_model*.safetensors",
    ])
    if dit:
        components["wan_video_dit"] = dit

    te = _first_existing_pattern(model_dir, [
        "models_t5_umt5-xxl-enc-bf16*.pth",
        "text_encoder/models_t5_umt5-xxl-enc-bf16*.pth",
        "text_encoder/model-*.safetensors",
        "text_encoder/model.safetensors",
    ])
    if te:
        components["wan_video_text_encoder"] = te

    vae = _first_existing_pattern(model_dir, [
        "Wan2.2_VAE*.pth",
        "Wan2.1_VAE*.pth",
        "vae/Wan2.2_VAE*.pth",
        "vae/Wan2.1_VAE*.pth",
        "vae/diffusion_pytorch_model*.safetensors",
    ])
    if vae:
        components["wan_video_vae"] = vae

    return components


def _expand_paths(model_dir: str, rel_path: str) -> list[str]:
    matches = glob.glob(os.path.join(model_dir, rel_path))
    matches.sort()
    return matches


def resolve_model_paths(model_dir: str, components: dict) -> dict:
    resolved = {}
    for name, rel_path in components.items():
        if isinstance(rel_path, list):
            expanded = []
            for item in rel_path:
                expanded.extend(_expand_paths(model_dir, item))
            resolved[name] = expanded
        else:
            expanded = _expand_paths(model_dir, rel_path)
            if not expanded:
                raise FileNotFoundError(f"Component {name} not found: {os.path.join(model_dir, rel_path)}")
            if glob.has_magic(rel_path):
                resolved[name] = expanded
            else:
                resolved[name] = expanded[0] if len(expanded) == 1 else expanded
    return resolved


def load_model_dir(model_dir: str) -> tuple[dict, dict]:
    model_index = load_model_index(model_dir) or {}
    components = model_index.get("components") or _infer_components(model_dir)
    if not components:
        raise FileNotFoundError(
            f"No model index found in {model_dir} and failed to infer components. "
            "Create diffsynth_model_index.json/model_index.json with a 'components' mapping."
        )
    resolved = resolve_model_paths(model_dir, components)
    return model_index, resolved
