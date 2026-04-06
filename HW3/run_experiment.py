#!/usr/bin/env python3
"""
Прогон набора промптов через Ollama (Llama / Mistral / Qwen / Gemma и т.д.).
Результат: CSV в папке results/ для последующей разметки и подсчёта метрик.

Запуск:
  cd HW3
  python3 run_experiment.py --models llama3,mistral
  python3 run_experiment.py --models llama3 --system-mode safe   # только безопасный ассистент
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROMPTS_PATH = ROOT / "prompts.json"
RESULTS_DIR = ROOT / "results"


def load_prompts(path: Path) -> tuple[list[tuple[str, str]], list[dict]]:
    """Возвращает список (system_profile_id, system_text) и промпты."""
    data = json.loads(path.read_text(encoding="utf-8"))
    prompts = data.get("prompts", [])
    profiles: list[tuple[str, str]] = []
    safe = (data.get("system_prompt") or "").strip()
    if safe:
        profiles.append(("safe", data.get("system_prompt", "")))
    plain = data.get("system_prompt_plain")
    if plain is not None and str(plain).strip():
        profiles.append(("plain", str(plain)))
    if not profiles:
        profiles = [("default", "")]
    return profiles, prompts


def ollama_chat(
    base_url: str,
    model: str,
    system: str,
    user: str,
    temperature: float,
    num_predict: int,
    timeout: int,
    num_ctx: int | None,
    num_thread: int | None,
) -> str:
    url = base_url.rstrip("/") + "/api/chat"
    options: dict = {"temperature": temperature, "num_predict": num_predict}
    if num_ctx is not None:
        options["num_ctx"] = num_ctx
    if num_thread is not None:
        options["num_thread"] = num_thread
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": options,
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode('utf-8', errors='replace')}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Не удалось подключиться к Ollama ({url}): {e}") from e

    msg = data.get("message") or {}
    return (msg.get("content") or "").strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="ДЗ3: прогон промптов через Ollama")
    parser.add_argument(
        "--models",
        default="llama3,mistral",
        help="Список моделей Ollama через запятую (должны быть ollama pull …)",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://127.0.0.1:11434",
        help="Базовый URL Ollama",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Temperature для всех вызовов (фиксируем для сравнения моделей)",
    )
    parser.add_argument(
        "--num-predict",
        type=int,
        default=384,
        help="Макс. токенов ответа (Ollama: num_predict). Меньше — быстрее и легче для RAM.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="Таймаут HTTP на один запрос, сек (8B на слабом Mac может идти минутами — не обрывать рано).",
    )
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=None,
        help="Опционально: сузить контекст Ollama (num_ctx), например 2048 — меньше памяти, стабильнее на 8B.",
    )
    parser.add_argument(
        "--num-thread",
        type=int,
        default=None,
        help="Опционально: число потоков CPU для Ollama (num_thread), на слабом Mac иногда помогает 4–6.",
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=PROMPTS_PATH,
        help="Путь к prompts.json",
    )
    parser.add_argument(
        "--system-mode",
        choices=("both", "safe", "plain"),
        default="both",
        help="Какой системный профиль из prompts.json прогонять: safe (безопасный ассистент), plain (без этой роли), both — оба подряд.",
    )
    args = parser.parse_args()

    if not args.prompts.is_file():
        print(f"Нет файла промптов: {args.prompts}", file=sys.stderr)
        return 1

    profiles, prompts = load_prompts(args.prompts)
    if args.system_mode == "safe":
        profiles = [p for p in profiles if p[0] == "safe"]
    elif args.system_mode == "plain":
        profiles = [p for p in profiles if p[0] == "plain"]
    if not profiles:
        print("Нет системных профилей для выбранного --system-mode.", file=sys.stderr)
        return 1

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        print("Укажите хотя бы одну модель: --models llama3", file=sys.stderr)
        return 1

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"run_{ts}.csv"

    fieldnames = [
        "timestamp_utc",
        "model",
        "system_profile",
        "prompt_id",
        "category",
        "user_prompt",
        "system_prompt_hash",
        "temperature",
        "num_predict",
        "num_ctx",
        "num_thread",
        "response",
        "error",
        "latency_sec",
    ]

    rows_written = 0
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()

        for model in models:
            for profile_id, system_text in profiles:
                sys_hash = str(hash(system_text) & 0xFFFFFFFF)
                for p in prompts:
                    pid = p.get("id", "")
                    cat = p.get("category", "")
                    text = p.get("text", "")
                    print(f"→ {model}  [{profile_id}]  {pid}  ({cat}) …", flush=True)
                    t0 = time.perf_counter()
                    err = ""
                    resp_text = ""
                    try:
                        resp_text = ollama_chat(
                            args.ollama_url,
                            model,
                            system_text,
                            text,
                            args.temperature,
                            args.num_predict,
                            args.timeout,
                            args.num_ctx,
                            args.num_thread,
                        )
                    except Exception as e:  # noqa: BLE001
                        err = str(e)
                    latency = time.perf_counter() - t0

                    writer.writerow(
                        {
                            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                            "model": model,
                            "system_profile": profile_id,
                            "prompt_id": pid,
                            "category": cat,
                            "user_prompt": text,
                            "system_prompt_hash": sys_hash,
                            "temperature": args.temperature,
                            "num_predict": args.num_predict,
                            "num_ctx": args.num_ctx if args.num_ctx is not None else "",
                            "num_thread": args.num_thread if args.num_thread is not None else "",
                            "response": resp_text,
                            "error": err,
                            "latency_sec": f"{latency:.3f}",
                        }
                    )
                    rows_written += 1
                    f.flush()

    print(f"Готово: {rows_written} строк → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
