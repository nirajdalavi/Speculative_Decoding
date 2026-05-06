# Speculative Decoding Across Two GPUs

This repository contains a runnable implementation of baseline greedy decoding and simplified speculative decoding using Hugging Face Transformers + PyTorch.

## Setup

```bash
python -m pip install torch transformers
```

Use a machine with at least two CUDA GPUs for the intended run:

- Draft model on `cuda:0`
- Target model on `cuda:1`


## Run

Requires **Linux (or Windows with CUDA) + two NVIDIA GPUs**. macOS is not supported by this script.

```bash
python main.py
```

If CUDA is missing or fewer than two GPUs are visible, the script exits with an error.

Optional flags:

```bash
python main.py --k-values 2 4 8 --max-new-tokens 100 --num-prompts 10
```

The script benchmarks:

1. Baseline greedy decoding on the target model
2. Speculative decoding for `k=2`, `k=4`, and `k=8`

It prints a result table and writes `benchmark_results.json`.

## Design Decisions

- Used `distilgpt2` as draft and `gpt2` as target for a simple, assignment-aligned pair.
- Explicit multi-GPU placement: draft on `cuda:0`, target on `cuda:1`.
- Both decoding paths use KV-cache (`use_cache=True`) to avoid recomputing full context each step.
- A short warmup run before timing to reduce one-time kernel startup noise.
- Speculative runs report **acceptance rate** (draft proposals that match the target’s greedy next token) to interpret throughput.

## When Speculative Helps

Speculative decoding speeds up decoding when all of the following are true:

- The draft model predicts the target model's next tokens accurately (high acceptance).
- The draft model is much cheaper/faster than the target model.
- Implementation overhead is low (minimal transfer, synchronization, and control-flow overhead).

## Performance Without KV Cache (main.py)

| Method      | k | tok/s  | avg latency (ms) | speedup | acceptance | runtime (s) |
|-------------|---|-------:|-----------------:|--------:|------------|------------:|
| Greedy      | — | 270.70 | 3.69             | 1.00×   | —          | 3.69        |
| Speculative | 2 | 321.42 | 3.11             | 1.19×   | 76.6%      | 3.11        |
| Speculative | 4 | 305.17 | 3.28             | 1.13×   | 65.1%      | 3.28        |
| Speculative | 8 | 238.79 | 4.19             | 0.88×   | 46.8%      | 4.19        |

Here speculative can **beat greedy for moderate `k`** because verifier work is amortized across a chunk in one forward, while greedy pays a growing full-attention cost each token; **`k=8`** still loses ground as acceptance (~46.8%) and chunk overhead dominate.

### Bottlenecks (`main.py`)

1. **Greedy: repeated full-context target attention every step** — no KV reuse, so cost rises badly with decode length compared to incremental decode.
2. **Speculative draft phase:** sequential **full-context** draft forwards (**`k`** times per verifier round); the draft model is smaller, but this adds steady per-round overhead.
3. **Cross-GPU coupling:** tensor copies/`ctx` moves between GPUs and **`torch.cuda.synchronize`** around timing amplify overhead on short steps.
4. **Large `k` pathologies:** long proposal strips mean **more wasted verifier work** when the draft diverges early; throughput does **not** scale linearly with **`k`** once acceptance drops.
5. **Control-flow overhead:** tight Python drafting/verification loops vs fewer, fatter fused GPU steps.

### Improvements (`main.py`)

1. **KV cache on greedy** (`use_cache=True`, feed only the latest token once primed) — removes the dominant repeated full-attention cost on the verifier baseline.
2. **KV cache on the draft speculative loop** — avoids **`k` full-context draft forwards** per round.
3. **`k` tuning from measured acceptance** — keep **`k`** where acceptance stays high enough that chunked verification wins.
4. **Fewer syncs/copies** — sync only where correctness requires it; reuse buffers; avoid shuttling full verifier context to the draft GPU unless the algorithm demands it.
5. **Numerics/dtype** — e.g. FP16/BF16 on CUDA where acceptable, especially when attention is bandwidth-bound.
6. **Adaptive chunk length** — cap effective proposal length from rolling acceptance stats so you don’t pay large **`k`** when empirical acceptance is low.

## Performance Results with KV cache (main2.py)

Method      | k | tok/s  | avg latency (ms) | speedup | acceptance | runtime (s) |
------------|---|-------:|-----------------:|--------:|-----------:|------------:|
Greedy      | - | 320.37 | 3.12             | 1.00x   | -          | 3.12        |
Speculative | 2 | 146.53 | 6.82             | 0.46x   | 66.1%      | 6.82        |
Speculative | 4 | 112.62 | 8.88             | 0.35x   | 37.6%      | 8.88        |
Speculative | 8 | 72.86  | 13.73            | 0.23x   | 19.3%      | 13.73       |


## After slight optimizations (main3.py)

Method      | k | tok/s  | avg latency (ms) | speedup | acceptance | runtime (s) |
------------|---|-------:|-----------------:|--------:|-----------:|------------:|
Greedy      | - | 313.44 | 3.19             | 1.00x   | -          | 3.19        |
Speculative | 2 | 168.39 | 5.94             | 0.54x   | 84.6%      | 5.94        |
Speculative | 4 | 140.25 | 7.13             | 0.45x   | 59.3%      | 7.13        |
Speculative | 8 | 166.74 | 6.00             | 0.53x   | 48.6%      | 6.00        |

