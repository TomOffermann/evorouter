"""Milestone 1: evaluate the base model on ARC and measure its router statistics.

Reports acc / acc_norm / fitness per split (compare test acc_norm with C3PO's 51.3 and with
lm-eval-harness), evaluator throughput, and per-layer top-k margins and loads on a search set
(the margins set the CMA-ES step size).

    python scripts/eval_base.py --splits test validation --out base.json          # cluster
    python scripts/eval_base.py --tiny --out /tmp/base.json                       # CPU, offline
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from evorouter.diagnostics import expert_loads, router_logits_at_scored_positions, topk_margins
from evorouter.evaluate import Evaluator
from evorouter.genome import BiasGenome
from evorouter.models import OLMOE_ID, byte_encode, hf_encoder, load_model, tiny_olmoe
from evorouter.routing import last_layers
from evorouter.runinfo import env_info
from evorouter.tasks.base import MCQuestion, search_val_split

TINY_QUESTIONS = [
    MCQuestion("t1", "Which is renewable?", ("coal", "wind", "oil", "gas"), 1),
    MCQuestion("t2", "What do plants need?", ("light", "sand", "rock"), 0),
    MCQuestion("t3", "Largest planet?", ("Mars", "Venus", "Jupiter", "Earth", "Pluto"), 2),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", default=OLMOE_ID)
    ap.add_argument("--subset", default="ARC-Challenge")
    ap.add_argument("--splits", nargs="+", default=["test"], choices=["train", "validation", "test"])
    ap.add_argument("--limit", type=int, default=None, help="first N questions per split (debugging)")
    ap.add_argument("--adapted-layers", type=int, default=4, help="last N MoE layers for router statistics")
    ap.add_argument("--n-search", type=int, default=64)
    ap.add_argument("--search-seed", type=int, default=0)
    ap.add_argument("--max-batch-tokens", type=int, default=16384)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--tiny", action="store_true", help="random tiny OLMoE + synthetic questions, CPU, offline"
    )
    args = ap.parse_args()

    t0 = time.perf_counter()
    if args.tiny:
        model, encode = tiny_olmoe(), byte_encode
        data = {split: TINY_QUESTIONS for split in ("train", "validation", "test")}
    else:
        from evorouter.tasks.arc import load_arc

        model, tokenizer = load_model(args.model_id, device="cuda" if torch.cuda.is_available() else "cpu")
        encode = hf_encoder(tokenizer)
        data = load_arc(args.subset)
    load_seconds = time.perf_counter() - t0

    result: dict = {
        "args": {k: str(v) for k, v in vars(args).items()},
        "env": env_info(),
        "load_seconds": round(load_seconds, 1),
        "splits": {},
    }

    for split in args.splits:
        questions = data[split][: args.limit]
        ev = Evaluator(model, questions, encode, max_batch_tokens=args.max_batch_tokens)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t = time.perf_counter()
        scores = ev.evaluate(None)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        seconds = time.perf_counter() - t
        result["splits"][split] = {
            **scores.member(0),
            "n_questions": len(questions),
            "n_requests": len(ev.requests),
            "tokens": ev.num_tokens,
            "seconds": round(seconds, 2),
            "tokens_per_sec": round(ev.num_tokens / seconds),
        }
        print(f"[{split}] {json.dumps(result['splits'][split])}", flush=True)

    # Router statistics on a search set drawn exactly as a CMA-ES run would draw it.
    search, _ = search_val_split(data["train"], args.search_seed, min(args.n_search, len(data["train"])), 0)
    layers = last_layers(model, args.adapted_layers)
    genome = BiasGenome(layers, model.config.num_experts)
    ev = Evaluator(model, search, encode, genome, max_batch_tokens=args.max_batch_tokens)
    requests = ev.requests

    # On-GPU sanity check: a zero genome through our routed forward must reproduce the HF forward.
    diff = np.abs(ev.loglik(genome.zeros()[None]) - ev.loglik(None)).max()
    result["zero_genome_max_abs_diff"] = float(diff)
    print(f"[sanity] zero genome vs base: max |dLL| = {diff:.3e}", flush=True)
    logits = router_logits_at_scored_positions(model, requests, layers, args.max_batch_tokens)
    k = model.config.num_experts_per_tok
    result["router"] = {
        "layers_0based": list(layers),
        "topk_margin": topk_margins(logits, k),
        "loads": expert_loads(logits, k),
    }
    print(f"[router] {json.dumps(result['router'])}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
