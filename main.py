#!/usr/bin/env python3
"""Two-GPU autoregressive comparison: greedy target-only decode vs draft–verify speculative."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from functools import partial
from statistics import mean
from typing import List, Optional, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# -----------------------------------------------------------------------------
# Part 3: Multi-GPU placement & per-device tensors (draft cuda:0, target cuda:1).
# -----------------------------------------------------------------------------

# Separate hardware roles (draft vs verifier) keeps the workload explicit.
DEVICE_DRAFT = torch.device("cuda:0")
DEVICE_VERIFY = torch.device("cuda:1")

MODEL_DRAFT = "distilgpt2"
MODEL_TARGET = "gpt2"

# Ten short openers; wording is original to avoid overlap with reused benchmark lists elsewhere.
OPENERS: Sequence[str] = (
    "After years of radio silence, the probe finally",
    "The trial ended early because evidence showed",
    "Inside the cramped lab, no one predicted that",
    "Before electricity was common, cities relied on",
    "The glacier moved slower than rumor suggested, yet",
    "Investors shrugged once the regulators clarified",
    "Through the canyon, echoes carried further than",
    "The debugger stopped on a branch that never",
    "Folk stories insisted the lighthouse had been dark since",
    "When the rover crossed the ridge, telemetry showed",
)


class InferenceBundle:
    __slots__ = ("draft_lm", "verifier_lm", "tokenizer")

    def __init__(
        self,
        draft_lm: AutoModelForCausalLM,
        verifier_lm: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
    ) -> None:
        self.draft_lm = draft_lm
        self.verifier_lm = verifier_lm
        self.tokenizer = tokenizer


def _sync_pair() -> None:
    torch.cuda.synchronize(DEVICE_DRAFT)
    torch.cuda.synchronize(DEVICE_VERIFY)


def build_bundle() -> InferenceBundle:
    tok = AutoTokenizer.from_pretrained(MODEL_TARGET)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    draft = AutoModelForCausalLM.from_pretrained(MODEL_DRAFT)
    draft.to(DEVICE_DRAFT)
    draft.eval()

    verifier = AutoModelForCausalLM.from_pretrained(MODEL_TARGET)
    verifier.to(DEVICE_VERIFY)
    verifier.eval()

    return InferenceBundle(draft, verifier, tok)


def _encode_fragment(tok: AutoTokenizer, text: str, place_on: torch.device) -> torch.Tensor:
    chunk = tok(text, return_tensors="pt").input_ids
    return chunk.to(place_on)


# -----------------------------------------------------------------------------
# Part 1: Baseline autoregressive greedy decoding on the target model only.
# -----------------------------------------------------------------------------


@torch.inference_mode()
def sample_next_id_from_lm(
    language_model: AutoModelForCausalLM,
    context_ids: torch.Tensor,
) -> torch.Tensor:
    """Return shape [batch, 1] greedy token id extending `context_ids`."""
    forward = language_model(context_ids)
    last_row = forward.logits[:, -1, :]
    greedy = torch.argmax(last_row, dim=-1, keepdim=True)
    return greedy


@torch.inference_mode()
def autoregressive_greedy_on_verifier_gpu(
    bundle: InferenceBundle,
    text: str,
    new_tokens: int,
) -> torch.Tensor:
    """Extend the prompt greedily using only `bundle.verifier_lm` on DEVICE_VERIFY."""
    ctx = _encode_fragment(bundle.tokenizer, text, DEVICE_VERIFY)
    for _ in range(new_tokens):
        token = sample_next_id_from_lm(bundle.verifier_lm, ctx)
        ctx = torch.cat([ctx, token], dim=1)
    return ctx


# -----------------------------------------------------------------------------
# Part 2: Speculative decoding — draft proposals on draft GPU, batched verify on target.
# -----------------------------------------------------------------------------


@torch.inference_mode()
def speculative_rollout_two_chip(
    bundle: InferenceBundle,
    text: str,
    new_tokens: int,
    lookahead: int,
) -> Tuple[torch.Tensor, float]:
    """Draft `lookahead` candidates on DEVICE_DRAFT, verify with one large verifier forward on DEVICE_VERIFY."""

    verifier_ctx = _encode_fragment(bundle.tokenizer, text, DEVICE_VERIFY)
    drafted = 0
    draft_checks = 0
    draft_matches_prefix = 0

    while drafted < new_tokens:
        horizon = min(lookahead, new_tokens - drafted)
        if horizon == 0:
            break

        draft_ctx = verifier_ctx.to(DEVICE_DRAFT)
        drafts: List[torch.Tensor] = []
        for _ in range(horizon):
            nxt = sample_next_id_from_lm(bundle.draft_lm, draft_ctx)
            drafts.append(nxt)
            draft_ctx = torch.cat([draft_ctx, nxt], dim=1)

        proposal_strip = torch.cat(drafts, dim=1).to(DEVICE_VERIFY, non_blocking=True)
        verifier_input = torch.cat([verifier_ctx, proposal_strip], dim=1)
        verifier_out = bundle.verifier_lm(verifier_input).logits

        pref_len_before = verifier_ctx.shape[1]
        draft_checks += horizon
        agree = 0
        while agree < horizon:
            logit_slice = verifier_out[:, pref_len_before + agree - 1, :]
            verifier_pick = torch.argmax(logit_slice, dim=-1)
            if verifier_pick.item() == proposal_strip[0, agree].item():
                agree += 1
            else:
                break

        draft_matches_prefix += agree

        if agree > 0:
            verifier_ctx = torch.cat([verifier_ctx, proposal_strip[:, :agree]], dim=1)
            drafted += agree

        if agree < horizon:
            correction = torch.argmax(
                verifier_out[:, pref_len_before + agree - 1, :], dim=-1, keepdim=True
            )
            verifier_ctx = torch.cat([verifier_ctx, correction], dim=1)
            drafted += 1
        else:
            if drafted < new_tokens:
                follow_up = torch.argmax(
                    verifier_out[:, pref_len_before + horizon - 1, :], dim=-1, keepdim=True
                )
                verifier_ctx = torch.cat([verifier_ctx, follow_up], dim=1)
                drafted += 1

    prompt_len = _encode_fragment(bundle.tokenizer, text, DEVICE_VERIFY).shape[1]
    acceptance = (draft_matches_prefix / draft_checks) if draft_checks else 0.0
    return verifier_ctx[:, : prompt_len + new_tokens], acceptance


@dataclass
class MethodRow:
    method: str
    k: Optional[int]
    total_runtime_s: float
    total_tokens: int
    tokens_per_sec: float
    avg_latency_ms: float
    speedup_vs_greedy: float
    mean_acceptance_rate: Optional[float] = None


def print_comparison_table(rows: Sequence[MethodRow]) -> None:
    """Same grid style as ``main.py``."""
    headers = ["Method", "k", "tok/s", "avg latency (ms)", "speedup", "acceptance", "runtime (s)"]
    out_rows = []
    for r in rows:
        out_rows.append(
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

    widths = [max(len(str(row[i])) for row in ([headers] + out_rows)) for i in range(len(headers))]
    fmt = " | ".join("{:<" + str(w) + "}" for w in widths)
    sep = "-+-".join("-" * w for w in widths)
    print(fmt.format(*headers))
    print(sep)
    for row in out_rows:
        print(fmt.format(*row))


def _duration_token_totals_only_greedy(
    bundle: InferenceBundle,
    corpus: Sequence[str],
    max_new_tokens: int,
) -> Tuple[float, int]:
    wall = 0.0
    tokens = 0
    greedy_fn = partial(
        autoregressive_greedy_on_verifier_gpu,
        bundle,
        new_tokens=max_new_tokens,
    )
    for excerpt in corpus:
        _sync_pair()
        start = time.perf_counter()
        sequence = greedy_fn(text=excerpt)
        _sync_pair()
        wall += time.perf_counter() - start
        prefix = bundle.tokenizer(excerpt, return_tensors="pt").input_ids.shape[1]
        emitted = sequence.shape[1] - prefix
        tokens += emitted
    return wall, tokens


def _duration_token_totals_speculative(
    bundle: InferenceBundle,
    corpus: Sequence[str],
    max_new_tokens: int,
    lookahead: int,
) -> Tuple[float, int, Optional[float]]:
    wall = 0.0
    tokens = 0
    rates: List[float] = []
    spec_fn = partial(
        speculative_rollout_two_chip,
        bundle,
        new_tokens=max_new_tokens,
        lookahead=lookahead,
    )
    for excerpt in corpus:
        _sync_pair()
        start = time.perf_counter()
        sequence, accept_hint = spec_fn(text=excerpt)
        _sync_pair()
        wall += time.perf_counter() - start
        prefix = bundle.tokenizer(excerpt, return_tensors="pt").input_ids.shape[1]
        emitted = sequence.shape[1] - prefix
        tokens += emitted
        rates.append(accept_hint)
    return wall, tokens, mean(rates) if rates else None


def _quiet_warmup(bundle: InferenceBundle, preview_len: int = 4) -> None:
    first = OPENERS[0]
    _ = autoregressive_greedy_on_verifier_gpu(bundle, first, preview_len)
    _ = speculative_rollout_two_chip(bundle, first, preview_len, lookahead=2)[0]
    _sync_pair()


def _parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare speculative draft–verify rollout against greedy target decode (two GPUs).",
    )
    parser.add_argument("--max-new-tokens", type=int, default=100, dest="max_new")
    parser.add_argument(
        "--k-values",
        "--k-grid",
        type=int,
        nargs="+",
        default=[2, 4, 8],
        dest="k_values",
        help="Draft widths `k` to benchmark (aligned with assignment sweep).",
    )
    parser.add_argument("--warmup-pass", type=int, default=2, dest="warmup_pass")
    return parser.parse_args()


def driver() -> None:
    cli = _parse_cli()
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    if torch.cuda.device_count() < 2:
        print("CUDA devices found:", torch.cuda.device_count(), "(need ≥2).")
        return

    bundle = build_bundle()

    print(f"Draft device:  {DEVICE_DRAFT}")
    print(f"Target device: {DEVICE_VERIFY}")
    print("Running warmup...")
    for _ in range(cli.warmup_pass):
        _quiet_warmup(bundle)

    print("Running baseline greedy decoding...\n")
    greedy_wall, greedy_tokens = _duration_token_totals_only_greedy(bundle, OPENERS, cli.max_new)
    greedy_tok_s = greedy_tokens / greedy_wall
    greedy_lat_ms = (greedy_wall / greedy_tokens) * 1000.0

    table_rows: List[MethodRow] = [
        MethodRow(
            method="Greedy",
            k=None,
            total_runtime_s=greedy_wall,
            total_tokens=greedy_tokens,
            tokens_per_sec=greedy_tok_s,
            avg_latency_ms=greedy_lat_ms,
            speedup_vs_greedy=1.0,
            mean_acceptance_rate=None,
        )
    ]

    for cand_k in cli.k_values:
        print(f"Running speculative decoding with k={cand_k}...\n")
        spec_wall, spec_tokens, acceptance = _duration_token_totals_speculative(
            bundle,
            OPENERS,
            cli.max_new,
            cand_k,
        )
        spec_tok_s = spec_tokens / spec_wall if spec_wall > 0 else float("nan")
        spec_lat_ms = (spec_wall / spec_tokens) * 1000.0 if spec_tokens > 0 else float("nan")
        speedup = spec_tok_s / greedy_tok_s if greedy_tok_s > 0 else float("nan")
        table_rows.append(
            MethodRow(
                method="Speculative",
                k=cand_k,
                total_runtime_s=spec_wall,
                total_tokens=spec_tokens,
                tokens_per_sec=spec_tok_s,
                avg_latency_ms=spec_lat_ms,
                speedup_vs_greedy=speedup,
                mean_acceptance_rate=acceptance,
            )
        )

    print()
    print_comparison_table(table_rows)
    print()


if __name__ == "__main__":
    driver()
