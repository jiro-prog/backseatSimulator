"""
Tower layer sensitivity analysis (leave-one-out).

全層bf16ベースラインから、encoder layer単位で4bitに戻したときの
tower最終出力の cosine similarity 低下を測定する。

Usage:
    .venv/Scripts/python scripts/sensitivity_analysis.py --tower vision
    .venv/Scripts/python scripts/sensitivity_analysis.py --tower audio
    .venv/Scripts/python scripts/sensitivity_analysis.py --tower both
"""

import argparse
import io
import logging
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import bitsandbytes as bnb
import numpy as np
import scipy.io.wavfile
import torch
import yaml
from PIL import Image
from safetensors import safe_open
from huggingface_hub import snapshot_download

from ai.analyzer import load_model

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEBUG_DUMP = os.path.join(PROJECT_ROOT, "debug_dump")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config():
    with open(os.path.join(PROJECT_ROOT, "config.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)


def group_modules_by_block(module_names: list[str], tower_name: str) -> dict[str, list[str]]:
    """Module names -> {block_key: [module_names]}."""
    groups: dict[str, list[str]] = {}
    for name in module_names:
        if "vision_tower" in tower_name:
            if "patch_embedder" in name:
                key = "patch_embedder"
            else:
                m = re.search(r"encoder\.layers\.(\d+)\.", name)
                key = f"encoder.layers.{m.group(1)}" if m else "other"
        else:  # audio_tower
            if "subsample_conv_projection" in name:
                key = "subsample_conv_projection"
            elif name.endswith("output_proj"):
                key = "output_proj"
            else:
                m = re.search(r"layers\.(\d+)\.", name)
                key = f"layers.{m.group(1)}" if m else "other"
        groups.setdefault(key, []).append(name)
    return groups


def block_sort_key(key: str):
    if key in ("patch_embedder", "subsample_conv_projection"):
        return -1, key
    if key == "output_proj":
        return 9999, key
    m = re.search(r"(\d+)$", key)
    return (int(m.group(1)), key) if m else (0, key)


def swap_modules(model, names: list[str], source: dict):
    """Swap modules in model tree from *source* dict."""
    for name in names:
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], source[name])


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        a.flatten().float().unsqueeze(0),
        b.flatten().float().unsqueeze(0),
    ).item()


# ---------------------------------------------------------------------------
# Dequantization with ref-saving
# ---------------------------------------------------------------------------

def dequantize_and_save(model, model_id: str, tower_name: str):
    """
    Tower内の全 Linear4bit を bf16 に差し替えつつ、
    差し替え前(4bit)・差し替え後(bf16) 双方の参照を返す。
    """
    import glob as glob_mod

    quantized = {}
    for name, module in model.named_modules():
        if tower_name in name and isinstance(module, bnb.nn.Linear4bit):
            quantized[name] = module

    logger.info("%s: %d Linear4bit modules", tower_name, len(quantized))
    if not quantized:
        return {}, {}

    saved_4bit = dict(quantized)

    # safetensors から bf16 weight をロード
    model_path = snapshot_download(model_id, local_files_only=True)
    st_files = glob_mod.glob(os.path.join(model_path, "*.safetensors"))

    original_weights: dict[str, torch.Tensor] = {}
    for sf in st_files:
        with safe_open(sf, framework="pt", device="cpu") as f:
            for key in f.keys():
                if tower_name in key:
                    original_weights[key] = f.get_tensor(key)

    for module_name in quantized:
        wk = module_name + ".weight"
        bk = module_name + ".bias"
        if wk not in original_weights:
            logger.warning("weight not found: %s", wk)
            continue
        w = original_weights[wk].to(torch.bfloat16)
        b = original_weights.get(bk)
        if b is not None:
            b = b.to(torch.bfloat16)

        new_lin = torch.nn.Linear(
            w.shape[1], w.shape[0], bias=b is not None,
            dtype=torch.bfloat16, device="cuda:0",
        )
        new_lin.weight.data.copy_(w)
        if b is not None:
            new_lin.bias.data.copy_(b)

        parts = module_name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], new_lin)

    # bf16 参照を保存
    saved_bf16 = {}
    for name in quantized:
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        saved_bf16[name] = getattr(parent, parts[-1])

    logger.info("%s: dequantized %d modules  VRAM=%.0f MiB",
                tower_name, len(saved_bf16),
                torch.cuda.memory_allocated() / 1024**2)
    return saved_4bit, saved_bf16


# ---------------------------------------------------------------------------
# Input preparation
# ---------------------------------------------------------------------------

def prepare_vision_inputs(processor) -> list[tuple[str, dict]]:
    pil = Image.open(os.path.join(DEBUG_DUMP, "full.png"))
    messages = [{"role": "user", "content": [
        {"type": "image", "image": pil},
        {"type": "text", "text": "この画面を見てください。"},
    ]}]
    inp = processor.apply_chat_template(
        messages, add_generation_prompt=True,
        tokenize=True, return_dict=True, return_tensors="pt",
        enable_thinking=False,
    )
    return [("image", inp)]


def prepare_audio_inputs(processor) -> list[tuple[str, dict]]:
    pil = Image.open(os.path.join(DEBUG_DUMP, "full.png"))
    sr, raw = scipy.io.wavfile.read(os.path.join(DEBUG_DUMP, "audio.wav"))
    if raw.dtype == np.int16:
        audio_real = raw.astype(np.float32) / 32768.0
    else:
        audio_real = raw.astype(np.float32)

    # near-silence
    rng = np.random.default_rng(42)
    audio_silence = rng.normal(0, 0.001, size=sr * 5).astype(np.float32)

    inputs_list = []
    for label, audio in [("real_audio", audio_real), ("silence", audio_silence)]:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": pil},
            {"type": "audio", "audio": audio},
            {"type": "text", "text": "この画面と音声を確認してください。"},
        ]}]
        inp = processor.apply_chat_template(
            messages, add_generation_prompt=True,
            tokenize=True, return_dict=True, return_tensors="pt",
            enable_thinking=False,
        )
        inputs_list.append((label, inp))
    return inputs_list


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def run_analysis(model, processor, tower_name: str,
                 inputs_list: list[tuple[str, dict]], model_id: str) -> dict:
    print(f"\n{'='*80}")
    print(f"  Sensitivity Analysis: {tower_name}")
    print(f"{'='*80}")

    saved_4bit, saved_bf16 = dequantize_and_save(model, model_id, tower_name)
    if not saved_4bit:
        print("  No quantized layers, skipping.")
        return {}

    groups = group_modules_by_block(list(saved_4bit.keys()), tower_name)
    print(f"  Blocks: {len(groups)}  "
          f"({', '.join(k for k in sorted(groups, key=block_sort_key))})")

    # ---- forward hook ----
    captured: dict[str, torch.Tensor] = {}

    def hook_fn(_module, _inp, output):
        if isinstance(output, torch.Tensor):
            captured["out"] = output.detach()
        elif isinstance(output, tuple) and len(output) > 0:
            captured["out"] = output[0].detach()
        elif hasattr(output, "last_hidden_state"):
            captured["out"] = output.last_hidden_state.detach()
        else:
            captured["out"] = None
            logger.warning("Unexpected tower output type: %s", type(output))

    tower_mod = dict(model.named_modules()).get(f"model.{tower_name}")
    if tower_mod is None:
        print(f"  ERROR: model.{tower_name} not found in module tree")
        return {}
    hook = tower_mod.register_forward_hook(hook_fn)

    # ---- baseline (all bf16) ----
    print("\n  Computing baseline (all bf16)...")
    baselines: dict[str, torch.Tensor] = {}
    for label, inputs in inputs_list:
        dev = {k: v.to(model.device) if hasattr(v, "to") else v
               for k, v in inputs.items()}
        with torch.inference_mode():
            model(**dev)
        if captured.get("out") is None:
            print(f"  ERROR: hook did not capture output for {label}")
            hook.remove()
            return {}
        baselines[label] = captured["out"].clone()
        print(f"    {label}: shape={baselines[label].shape}")

    # ---- leave-one-out ----
    print(f"\n  {'block':<40s}  {'avg_cos':>9s}  ", end="")
    labels = [l for l, _ in inputs_list]
    print("  ".join(f"{l:>12s}" for l in labels), f"  {'mods':>4s}  {'MiB':>6s}")
    print(f"  {'-'*100}")

    results = {}
    total_t0 = time.time()
    for gk in sorted(groups, key=block_sort_key):
        names = groups[gk]
        swap_modules(model, names, saved_4bit)

        sims = {}
        for label, inputs in inputs_list:
            dev = {k: v.to(model.device) if hasattr(v, "to") else v
                   for k, v in inputs.items()}
            with torch.inference_mode():
                model(**dev)
            sims[label] = cosine_sim(captured["out"], baselines[label])

        swap_modules(model, names, saved_bf16)

        n_params = sum(
            saved_bf16[n].weight.numel()
            + (saved_bf16[n].bias.numel() if saved_bf16[n].bias is not None else 0)
            for n in names
        )
        bf16_mib = n_params * 2 / 1024**2
        avg = sum(sims.values()) / len(sims)
        results[gk] = dict(sims=sims, n_modules=len(names),
                           n_params=n_params, bf16_mib=bf16_mib)

        cols = "  ".join(f"{sims[l]:>12.8f}" for l in labels)
        print(f"  {gk:<40s}  {avg:>9.6f}  {cols}  {len(names):>4d}  {bf16_mib:>6.1f}")

    elapsed = time.time() - total_t0
    print(f"\n  Elapsed: {elapsed:.1f}s  ({elapsed / len(groups):.1f}s / block)")

    # ---- ranking ----
    print(f"\n  === Ranking (lower cos_sim = more sensitive) ===")
    ranked = sorted(results.items(),
                    key=lambda x: sum(x[1]["sims"].values()) / len(x[1]["sims"]))
    cumulative_mib = 0.0
    for rank, (gk, info) in enumerate(ranked, 1):
        avg = sum(info["sims"].values()) / len(info["sims"])
        cumulative_mib += info["bf16_mib"]
        cols = "  ".join(f"{info['sims'][l]:>12.8f}" for l in labels)
        print(f"  {rank:3d}. {gk:<40s}  avg={avg:.8f}  [{cols}]  "
              f"bf16={info['bf16_mib']:.1f} MiB  (cum={cumulative_mib:.1f})")

    hook.remove()

    # ---- 4bit refs 解放 ----
    del saved_4bit
    torch.cuda.empty_cache()
    logger.info("VRAM after cleanup: %.0f MiB", torch.cuda.memory_allocated() / 1024**2)

    return results


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Tower layer sensitivity analysis")
    parser.add_argument("--tower", choices=["vision", "audio", "both"], default="both")
    args = parser.parse_args()

    config = load_config()
    model_id = config.get("model_id", "google/gemma-4-E2B-it")

    # dequantize は自前で制御するので off
    run_config = dict(config)
    run_config["vision_fp16"] = False
    run_config["audio_fp16"] = False

    print(f"Loading model: {model_id} (4bit, no dequantize)")
    model, processor = load_model(run_config)
    print(f"VRAM after load: {torch.cuda.memory_allocated() / 1024**2:.0f} MiB\n")

    if args.tower in ("vision", "both"):
        inputs_list = prepare_vision_inputs(processor)
        run_analysis(model, processor, "vision_tower", inputs_list, model_id)

    if args.tower in ("audio", "both"):
        inputs_list = prepare_audio_inputs(processor)
        run_analysis(model, processor, "audio_tower", inputs_list, model_id)

    print("\n=== All done ===")


if __name__ == "__main__":
    main()
