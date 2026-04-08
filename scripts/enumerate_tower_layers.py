"""vision_tower / audio_tower 内の量子化Linear層を列挙し、パラメータ数を集計する。

使い方:
    .venv/Scripts/python scripts/enumerate_tower_layers.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml
import bitsandbytes as bnb
from transformers import AutoModelForMultimodalLM, AutoProcessor, BitsAndBytesConfig


def enumerate_layers(model, tower_name: str):
    """tower内のLinear4bit層を列挙して情報を返す。"""
    layers = []
    for name, module in model.named_modules():
        if tower_name not in name:
            continue
        if isinstance(module, bnb.nn.Linear4bit):
            in_f = module.in_features
            out_f = module.out_features
            has_bias = module.bias is not None
            # パラメータ数: weight + bias
            params = in_f * out_f + (out_f if has_bias else 0)
            layers.append({
                "name": name,
                "in_features": in_f,
                "out_features": out_f,
                "has_bias": has_bias,
                "params": params,
            })
    return layers


def format_bytes(n_bytes: int) -> str:
    if n_bytes >= 1024**3:
        return f"{n_bytes / 1024**3:.2f} GiB"
    return f"{n_bytes / 1024**2:.1f} MiB"


def print_tower_report(tower_name: str, layers: list):
    print(f"\n{'='*80}")
    print(f"  {tower_name}  —  {len(layers)} Linear4bit layers")
    print(f"{'='*80}")
    print(f"{'#':>4}  {'in':>6} × {'out':<6}  {'bias':>4}  {'params':>12}  module name")
    print(f"{'-'*80}")

    total_params = 0
    for i, layer in enumerate(layers):
        total_params += layer["params"]
        print(
            f"{i:4d}  {layer['in_features']:6d} × {layer['out_features']:<6d}  "
            f"{'yes' if layer['has_bias'] else 'no':>4}  "
            f"{layer['params']:>12,}  {layer['name']}"
        )

    bf16_bytes = total_params * 2  # bf16 = 2 bytes per param
    four_bit_bytes = total_params // 2  # 4bit = 0.5 bytes per param

    print(f"{'-'*80}")
    print(f"  合計パラメータ数: {total_params:>14,}")
    print(f"  bf16サイズ (CPU RAM): {format_bytes(bf16_bytes):>10}")
    print(f"  4bitサイズ (VRAM):    {format_bytes(four_bit_bytes):>10}")
    print(f"  差分 (bf16 - 4bit):   {format_bytes(bf16_bytes - four_bit_bytes):>10}")
    print()


def setup_stdout_utf8():
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def main():
    setup_stdout_utf8()
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_id = config.get("model_id", "google/gemma-4-E2B-it")
    print(f"モデルロード中: {model_id} (4bit, dequantize前)")

    # 4bitでロード（dequantizeはしない）
    kwargs = {
        "torch_dtype": "auto",
        "device_map": "cuda:0",
        "quantization_config": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
    }
    model = AutoModelForMultimodalLM.from_pretrained(model_id, **kwargs)

    for tower_name in ["vision_tower", "audio_tower"]:
        layers = enumerate_layers(model, tower_name)
        print_tower_report(tower_name, layers)

    # 全体のLinear4bit数も参考に出す
    all_4bit = sum(1 for _, m in model.named_modules() if isinstance(m, bnb.nn.Linear4bit))
    print(f"モデル全体のLinear4bit層数: {all_4bit}")


if __name__ == "__main__":
    main()
