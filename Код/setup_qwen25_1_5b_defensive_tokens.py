#!/usr/bin/env python3
"""
Сборка Qwen2.5-1.5B-Instruct + 5 DefensiveTokens для самых лёгких локальных прогонов.

В defensivetokens.json авторов есть только Qwen2.5-7B. Для 1.5B векторы из статьи
не поставляются — те же 5 спецтокенов и chat template, что у 7B в setup.py;
строки эмбеддинга для новых токенов — шум в масштабе std базовых эмбеддингов.

Запуск (из каталога Проект, venv DefensiveToken-main/defensivetoken):
  ../DefensiveToken-main/defensivetoken/bin/python setup_qwen25_1_5b_defensive_tokens.py

Выход по умолчанию:
  ../DefensiveToken-main/Qwen/Qwen2.5-1.5B-Instruct-5DefensiveTokens
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFENSIVE_TOKEN_NAMES = [
    "<|reserved_special_token_0|>",
    "<|reserved_special_token_1|>",
    "<|reserved_special_token_2|>",
    "<|reserved_special_token_3|>",
    "<|reserved_special_token_4|>",
]

QWEN25_CHAT_TEMPLATE = "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Qwen2.5-1.5B + 5 DefensiveTokens (random init embeddings)"
    )
    parser.add_argument(
        "--hf_model",
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Идентификатор модели на Hugging Face",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        default=None,
        help="Куда сохранить (по умолчанию: DefensiveToken-main/Qwen/Qwen2.5-1.5B-Instruct-5DefensiveTokens)",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_out = os.path.normpath(
        os.path.join(
            script_dir,
            "..",
            "DefensiveToken-main",
            "Qwen",
            "Qwen2.5-1.5B-Instruct-5DefensiveTokens",
        )
    )
    output_dir = os.path.abspath(args.output_dir or default_out)

    def log(msg: str) -> None:
        print(msg, flush=True)

    log(f"[info] base model: {args.hf_model}")
    log(f"[info] output_dir: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    log("[info] loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, trust_remote_code=True)

    num_added = tokenizer.add_special_tokens(
        {"additional_special_tokens": DEFENSIVE_TOKEN_NAMES}
    )
    if num_added != len(DEFENSIVE_TOKEN_NAMES):
        print(
            f"[warn] expected to add {len(DEFENSIVE_TOKEN_NAMES)} tokens, added {num_added}",
            file=sys.stderr,
        )

    tokenizer.chat_template = QWEN25_CHAT_TEMPLATE

    log("[info] loading model (may download weights)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_model,
        dtype=torch.float16,
        device_map=None,
        trust_remote_code=True,
    )

    model.resize_token_embeddings(len(tokenizer))

    emb = model.get_input_embeddings()
    w = emb.weight.data
    std = float(w.std().item()) if w.numel() > 0 else 0.02
    n_new = len(DEFENSIVE_TOKEN_NAMES)
    start = w.shape[0] - n_new
    if start < 0:
        raise RuntimeError("embedding table smaller than n_new tokens")
    noise = torch.randn(n_new, w.shape[1], dtype=w.dtype, device=w.device) * std
    w[start : start + n_new].copy_(noise)
    log(f"[info] initialized last {n_new} embedding rows with N(0, std≈{std:.6f})")

    log("[info] saving tokenizer + model...")
    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)

    ref_gen = os.path.normpath(
        os.path.join(
            script_dir,
            "..",
            "DefensiveToken-main",
            "Qwen",
            "Qwen2.5-7B-Instruct-5DefensiveTokens",
            "generation_config.json",
        )
    )
    if os.path.isfile(ref_gen):
        shutil.copy2(ref_gen, os.path.join(output_dir, "generation_config.json"))
        log("[info] copied generation_config.json from 7B build")

    log("[done] готово. В step3 укажи:")
    log(f"  --model {output_dir}")


if __name__ == "__main__":
    main()
