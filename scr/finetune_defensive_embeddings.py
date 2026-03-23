#!/usr/bin/env python3
"""
Дообучение только строк эмбеддингов [DefensiveToken0]…[DefensiveToken4] в модели *-5DefensiveTokens.

Формат JSONL (как в defensive-tuning): каждая строка — JSON с полями:
  instruction (str), data (str, можно ""), base_response (str)

Требуется GPU (CUDA) для Qwen-7B; на CPU только микро-прогон (--max_samples 8).

После обучения сохраняется чекпоинт с весами 5 токенов; подставить в модель:
  emb = model.get_input_embeddings()
  ckpt = torch.load("...", map_location="cpu")
  for tid, row in zip(ckpt["token_ids"], ckpt["rows"]):
      emb.weight.data[tid] = row.to(emb.weight.device, dtype=emb.weight.dtype)

Затем сохранить модель: model.save_pretrained(...)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
import transformers

from device_utils import resolve_device, resolve_dtype


def load_jsonl(path: Path, max_samples: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if len(rows) >= max_samples:
                break
    return rows


def build_prompt(
    tokenizer: transformers.PreTrainedTokenizer,
    instruction: str,
    data: str,
    *,
    add_defensive_tokens: bool,
) -> str:
    messages = [{"role": "system", "content": instruction}]
    if data and data.strip():
        messages.append({"role": "user", "content": data})
    else:
        messages.append({"role": "user", "content": "."})
    kwargs = dict(conversation=messages, tokenize=False, add_generation_prompt=True)
    try:
        return tokenizer.apply_chat_template(add_defensive_tokens=add_defensive_tokens, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(**kwargs)


def defensive_token_ids(tokenizer: transformers.PreTrainedTokenizer) -> List[int]:
    ids: List[int] = []
    for i in range(5):
        tok = f"[DefensiveToken{i}]"
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid == tokenizer.unk_token_id:
            raise RuntimeError(f"Token {tok!r} not in vocabulary — use *-5DefensiveTokens from setup.py")
        ids.append(tid)
    return ids


def zero_grad_except_rows(grad: torch.Tensor, keep_rows: List[int]) -> torch.Tensor:
    if grad is None:
        return grad
    g = grad.clone()
    keep = set(keep_rows)
    for i in range(g.shape[0]):
        if i not in keep:
            g[i].zero_()
    return g


def main() -> None:
    p = argparse.ArgumentParser(description="SGD только по эмбеддингам DefensiveTokens")
    p.add_argument(
        "--model",
        required=True,
        help="Путь к …-Instruct-5DefensiveTokens (например Qwen2.5-7B из setup.py или Qwen2.5-1.5B из setup_qwen25_1_5b_defensive_tokens.py)",
    )
    p.add_argument("--data", required=True, type=Path, help="JSONL: instruction, data, base_response")
    p.add_argument("--output", type=Path, default=Path("finetuned_defensive_rows.pt"))
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--max_samples", type=int, default=256)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    args = p.parse_args()

    device = resolve_device(args.device, no_cuda=False)
    dtype = resolve_dtype(args.dtype, device)
    print(f"[info] device={device} dtype={dtype}", flush=True)

    tok = transformers.AutoTokenizer.from_pretrained(args.model)
    try:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            args.model, dtype=dtype, low_cpu_mem_usage=True
        ).to(device)
    except TypeError:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=dtype, low_cpu_mem_usage=True
        ).to(device)
    model.train()

    for param in model.parameters():
        param.requires_grad = False

    emb = model.get_input_embeddings()
    emb.weight.requires_grad = True

    def_ids = defensive_token_ids(tok)

    def _hook(grad):
        return zero_grad_except_rows(grad, def_ids)

    h = emb.weight.register_hook(_hook)

    samples = load_jsonl(args.data, args.max_samples)
    if not samples:
        raise SystemExit("No samples in data file")

    opt = torch.optim.SGD([emb.weight], lr=args.lr)

    for ep in range(args.epochs):
        total_loss = 0.0
        for i, row in enumerate(samples):
            instruction = row.get("instruction", "")
            data = row.get("data") or ""
            target = row.get("base_response", "")
            prompt = build_prompt(tok, instruction, data, add_defensive_tokens=True)
            full = prompt + target + (tok.eos_token or "")
            enc = tok(full, max_length=args.max_length, truncation=True, return_tensors="pt")
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
            prompt_len = min(len(prompt_ids), input_ids.shape[1])

            labels = input_ids.clone()
            labels[:, :prompt_len] = -100
            labels[attention_mask == 0] = -100

            opt.zero_grad()
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = out.loss
            if loss is None or torch.isnan(loss):
                continue
            loss.backward()
            opt.step()

            total_loss += float(loss.detach())
            if (i + 1) % 50 == 0:
                print(f"[info] epoch {ep+1} sample {i+1}/{len(samples)} loss={loss.item():.4f}", flush=True)

        print(f"[info] epoch {ep+1} mean_loss={total_loss / max(len(samples), 1):.4f}", flush=True)

    h.remove()

    rows = emb.weight.data[def_ids].detach().cpu()
    torch.save({"token_ids": def_ids, "rows": rows, "model_path": str(args.model)}, args.output)
    print(f"[info] Saved defensive rows to {args.output}", flush=True)


if __name__ == "__main__":
    main()
