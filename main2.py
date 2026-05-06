#!/usr/bin/env python3
"""Benchmark greedy vs speculative decoding across two GPUs.

Assignment target:
- draft model: distilgpt2 on cuda:0
- target model: gpt2 on cuda:1

By default two CUDA GPUs are required. Use `--smoke-test` on machines without CUDA
(for example macOS) to run a minimal CPU sanity check — not valid for benchmarking.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import List, Optional, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_PROMPTS: List[str] = [
    "Once upon a time",
    "The meaning of life is",
    "In a surprising turn of events,",
    "Machine learning models",
    "At the edge of the universe,",
    "The chef placed",
    "During the experiment,",
    "Across the city,",
    "A small robot discovered",
    "Five years from now,",
]

# Assignment mapping:
# - Part 1: Baseline greedy decoding -> greedy_generate_one_prompt
# - Part 2: Speculative decoding -> speculative_generate_one_prompt
# - Part 3: Multi-GPU placement/transfer -> pick_devices + main model loading + per-device tensors
# - Part 4: Benchmark/reporting -> aggregate_results + print_table + save_results + main loop


@dataclass
class PromptStats:
    prompt: str
    generated_tokens: int
    runtime_s: float
    tokens_per_sec: float
    proposed_tokens: Optional[int] = None
    accepted_tokens: Optional[int] = None
    acceptance_rate: Optional[float] = None


@dataclass
class MethodStats:
    method: str
    k: Optional[int]
    total_runtime_s: float
    total_tokens: int
    tokens_per_sec: float
    avg_latency_ms: float
    speedup_vs_greedy: float
    mean_acceptance_rate: Optional[float] = None
    proposed_tokens: Optional[int] = None
    accepted_tokens: Optional[int] = None


@dataclass
class RunConfig:
    draft_model: str
    target_model: str
    max_new_tokens: int
    ks: List[int]
    prompts: List[str]
    seed: int
    output_json: str
    smoke_test: bool


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(description="Benchmark greedy vs speculative decoding.")
    parser.add_argument("--draft-model", default="distilgpt2")
    parser.add_argument("--target-model", default="gpt2")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--k-values", type=int, nargs="*", default=[2, 4, 8])
    parser.add_argument("--output-json", default="benchmark_results.json")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-prompts", type=int, default=10)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Tiny CPU-only run when CUDA is unavailable (sanity check, not for assignment numbers).",
    )
    args = parser.parse_args()

    prompts = DEFAULT_PROMPTS[: args.num_prompts]
    if len(prompts) < args.num_prompts:
        raise ValueError("Requested more prompts than are available in the built-in prompt list.")

    if args.smoke_test:
        prompts = prompts[: min(2, len(prompts))]
        max_new_tokens = min(args.max_new_tokens, 8)
        ks = list(dict.fromkeys(args.k_values or [4]))[:3] or [4]
    else:
        max_new_tokens = args.max_new_tokens
        ks = list(dict.fromkeys(args.k_values))

    return RunConfig(
        draft_model=args.draft_model,
        target_model=args.target_model,
        max_new_tokens=max_new_tokens,
        ks=ks,
        prompts=prompts,
        seed=args.seed,
        output_json=args.output_json,
        smoke_test=args.smoke_test,
    )


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_devices(*, smoke_test: bool = False) -> Tuple[torch.device, torch.device]:
    """Return devices for draft and target models.

    Default (assignment path): cuda:0 for draft and cuda:1 for target — two GPUs required.

    With smoke_test=True: both models live on CPU so you can sanity-check decoding on Mac.
    """
    if smoke_test:
        cpu = torch.device("cpu")
        return cpu, cpu

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. This script requires two CUDA GPUs. "
            "On macOS run: python main.py --smoke-test"
        )
    if torch.cuda.device_count() < 2:
        raise RuntimeError(
            f"Found {torch.cuda.device_count()} CUDA device(s), but this script requires at least 2."
        )
    return torch.device("cuda:0"), torch.device("cuda:1")


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_tokenizer(model_name: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(model_name: str, device: torch.device, dtype: torch.dtype) -> AutoModelForCausalLM:
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model.eval()
    model.to(device)
    return model


def step_model(
    model: AutoModelForCausalLM,
    token_ids: torch.Tensor,
    past_key_values,
) -> Tuple[object, torch.Tensor]:
    out = model(input_ids=token_ids, past_key_values=past_key_values, use_cache=True)
    return out.past_key_values, out.logits[:, -1, :]


@torch.inference_mode()
def greedy_generate_one_prompt(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
) -> PromptStats:
    # Part 1: Baseline autoregressive greedy decoding on target model only.
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)

    sync_device(device)
    start = time.perf_counter()

    out = model(input_ids=input_ids, use_cache=True)
    past = out.past_key_values
    next_logits = out.logits[:, -1, :]

    generated = 0
    while generated < max_new_tokens:
        next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
        generated += 1
        if generated == max_new_tokens:
            break
        past, next_logits = step_model(model, next_token, past)

    sync_device(device)
    runtime = time.perf_counter() - start
    return PromptStats(
        prompt=prompt,
        generated_tokens=max_new_tokens,
        runtime_s=runtime,
        tokens_per_sec=max_new_tokens / runtime if runtime > 0 else math.inf,
    )


@torch.inference_mode()
def speculative_generate_one_prompt(
    draft_model: AutoModelForCausalLM,
    target_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    draft_device: torch.device,
    target_device: torch.device,
    max_new_tokens: int,
    k: int,
) -> PromptStats:
    # Part 2: Simplified speculative decoding with draft proposal + target verification.
    inputs = tokenizer(prompt, return_tensors="pt")
    draft_input_ids = inputs["input_ids"].to(draft_device)
    target_input_ids = inputs["input_ids"].to(target_device)

    sync_device(draft_device)
    sync_device(target_device)
    start = time.perf_counter()

    draft_out = draft_model(input_ids=draft_input_ids, use_cache=True)
    draft_past = draft_out.past_key_values
    draft_logits = draft_out.logits[:, -1, :]

    target_out = target_model(input_ids=target_input_ids, use_cache=True)
    target_past = target_out.past_key_values
    target_logits = target_out.logits[:, -1, :]

    generated = 0
    proposed_total = 0
    accepted_total = 0

    while generated < max_new_tokens:
        remaining = max_new_tokens - generated
        n_propose = min(k, remaining)

        proposed_tokens: List[torch.Tensor] = []
        draft_states: List[object] = [draft_past]
        cur_past = draft_past
        cur_logits = draft_logits

        for _ in range(n_propose):
            token = torch.argmax(cur_logits, dim=-1, keepdim=True)
            proposed_tokens.append(token)
            proposed_total += 1
            cur_past, cur_logits = step_model(draft_model, token, cur_past)
            draft_states.append(cur_past)

        verify_past = target_past
        verify_logits = target_logits
        accepted_prefix = 0
        mismatch_token: Optional[torch.Tensor] = None

        for proposed in proposed_tokens:
            target_next = torch.argmax(verify_logits, dim=-1, keepdim=True)
            # Token ids are integers; compare without requiring same GPU tensor placement.
            if proposed.item() == target_next.item():
                accepted_prefix += 1
                accepted_total += 1
                generated += 1
                token_on_target = proposed.to(target_device, non_blocking=True)
                verify_past, verify_logits = step_model(target_model, token_on_target, verify_past)
                continue
            mismatch_token = target_next
            break

        if mismatch_token is None:
            draft_past = draft_states[accepted_prefix]
            draft_logits = cur_logits
            target_past = verify_past
            target_logits = verify_logits
            continue

        generated += 1
        accepted_total += 1
        draft_past = draft_states[accepted_prefix]
        draft_past, draft_logits = step_model(
            draft_model, mismatch_token.to(draft_device, non_blocking=True), draft_past
        )
        target_past, target_logits = step_model(target_model, mismatch_token, verify_past)

    sync_device(draft_device)
    sync_device(target_device)
    runtime = time.perf_counter() - start
    return PromptStats(
        prompt=prompt,
        generated_tokens=max_new_tokens,
        runtime_s=runtime,
        tokens_per_sec=max_new_tokens / runtime if runtime > 0 else math.inf,
        proposed_tokens=proposed_total,
        accepted_tokens=accepted_total,
        acceptance_rate=(accepted_total / proposed_total) if proposed_total > 0 else None,
    )


def aggregate_results(
    method: str,
    k: Optional[int],
    prompt_results: Sequence[PromptStats],
    greedy_tokens_per_sec: Optional[float] = None,
) -> MethodStats:
    total_runtime = sum(r.runtime_s for r in prompt_results)
    total_tokens = sum(r.generated_tokens for r in prompt_results)
    tokens_per_sec = total_tokens / total_runtime if total_runtime > 0 else math.inf
    avg_latency_ms = (total_runtime / total_tokens) * 1000.0 if total_tokens > 0 else math.inf
    speedup = tokens_per_sec / greedy_tokens_per_sec if greedy_tokens_per_sec else 1.0

    proposed_tokens = sum(r.proposed_tokens or 0 for r in prompt_results) or None
    accepted_tokens = sum(r.accepted_tokens or 0 for r in prompt_results) or None
    rates = [r.acceptance_rate for r in prompt_results if r.acceptance_rate is not None]
    mean_acceptance_rate = mean(rates) if rates else None

    return MethodStats(
        method=method,
        k=k,
        total_runtime_s=total_runtime,
        total_tokens=total_tokens,
        tokens_per_sec=tokens_per_sec,
        avg_latency_ms=avg_latency_ms,
        speedup_vs_greedy=speedup,
        mean_acceptance_rate=mean_acceptance_rate,
        proposed_tokens=proposed_tokens,
        accepted_tokens=accepted_tokens,
    )


def print_table(results: Sequence[MethodStats]) -> None:
    headers = ["Method", "k", "tok/s", "avg latency (ms)", "speedup", "acceptance", "runtime (s)"]
    rows = []
    for r in results:
        rows.append(
            [
                r.method,
                "-" if r.k is None else str(r.k),
                f"{r.tokens_per_sec:.2f}",
                f"{r.avg_latency_ms:.2f}",
                f"{r.speedup_vs_greedy:.2f}x",
                "-" if r.mean_acceptance_rate is None else f"{100.0 * r.mean_acceptance_rate:.1f}%",
                f"{r.total_runtime_s:.2f}",
            ]
        )

    widths = [max(len(str(row[i])) for row in ([headers] + rows)) for i in range(len(headers))]
    fmt = " | ".join("{:<" + str(w) + "}" for w in widths)
    sep = "-+-".join("-" * w for w in widths)
    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(*row))


def save_results(path: Path, config: RunConfig, device_info: dict, results: Sequence[MethodStats]) -> None:
    payload = {
        "config": asdict(config),
        "devices": device_info,
        "results": [asdict(r) for r in results],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def warmup(
    draft_model: AutoModelForCausalLM,
    target_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    draft_device: torch.device,
    target_device: torch.device,
) -> None:
    warmup_prompt = "Warmup prompt"
    _ = greedy_generate_one_prompt(target_model, tokenizer, warmup_prompt, target_device, max_new_tokens=4)
    _ = speculative_generate_one_prompt(
        draft_model,
        target_model,
        tokenizer,
        warmup_prompt,
        draft_device,
        target_device,
        max_new_tokens=4,
        k=2,
    )


def main() -> None:
    config = parse_args()
    set_global_seed(config.seed)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    draft_device, target_device = pick_devices(smoke_test=config.smoke_test)
    dtype = torch.float32 if config.smoke_test else torch.float16

    if config.smoke_test:
        print(
            "*** SMOKE TEST (CPU): not assignment-grade benchmark; dual-GPU CUDA run is unchanged. ***\n"
        )

    # Part 3: Explicit multi-GPU placement (cuda:0 / cuda:1), or CPU for smoke test only.
    tokenizer = load_tokenizer(config.target_model)
    draft_model = load_model(config.draft_model, draft_device, dtype)
    target_model = load_model(config.target_model, target_device, dtype)

    print(f"Draft device:  {draft_device}")
    print(f"Target device: {target_device}")
    print("Running warmup...")
    warmup(draft_model, target_model, tokenizer, draft_device, target_device)

    print("Running baseline greedy decoding...\n")
    # Part 4 (Benchmark): run baseline and capture metrics.
    greedy_prompt_results: List[PromptStats] = []
    for prompt in config.prompts:
        greedy_prompt_results.append(
            greedy_generate_one_prompt(
                model=target_model,
                tokenizer=tokenizer,
                prompt=prompt,
                device=target_device,
                max_new_tokens=config.max_new_tokens,
            )
        )

    greedy_agg = aggregate_results("Greedy", None, greedy_prompt_results)
    all_results: List[MethodStats] = [greedy_agg]

    for k in config.ks:
        print(f"Running speculative decoding with k={k}...\n")
        # Part 4 (Benchmark): run speculative decoding for each required k value.
        prompt_results: List[PromptStats] = []
        for prompt in config.prompts:
            prompt_results.append(
                speculative_generate_one_prompt(
                    draft_model=draft_model,
                    target_model=target_model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    draft_device=draft_device,
                    target_device=target_device,
                    max_new_tokens=config.max_new_tokens,
                    k=k,
                )
            )
        all_results.append(aggregate_results("Speculative", k, prompt_results, greedy_agg.tokens_per_sec))

    print()
    print_table(all_results)

    device_info = {
        "draft_device": str(draft_device),
        "target_device": str(target_device),
        "num_cuda_devices": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "dtype": str(dtype).replace("torch.", ""),
    }
    out_path = Path(config.output_json)
    save_results(out_path, config, device_info, all_results)
    print(f"\nSaved benchmark summary to {out_path.resolve()}")


if __name__ == "__main__":
    main()
