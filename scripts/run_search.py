"""v0 search: evolve a selection-only bias on the last MoE layers of OLMoE for ARC-Challenge.

Resumable: re-running with the same --run-dir continues from the saved state (config and margins
are read from the run directory; CLI search settings are ignored on resume).

    python scripts/run_search.py --run-dir runs/cma/seed0 --seed 0 --optimizer cma
    python scripts/run_search.py --tiny --run-dir /tmp/tiny --generations 3 --popsize 4   # CPU
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from evorouter.context import tiny_data
from evorouter.diagnostics import router_logits_at_scored_positions, topk_margins
from evorouter.evaluate import Evaluator
from evorouter.genome import BiasGenome
from evorouter.models import OLMOE_ID, byte_encode, hf_encoder, load_model, tiny_olmoe
from evorouter.routing import last_layers
from evorouter.runinfo import env_info
from evorouter.search import SearchConfig, SearchRun
from evorouter.tasks.base import search_val_split


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--optimizer", choices=["cma", "sep-cma", "random"], default="cma")
    ap.add_argument("--popsize", type=int, default=64)
    ap.add_argument("--sigma0", type=float, default=1.0)
    ap.add_argument("--generations", type=int, default=200)
    ap.add_argument("--val-every", type=int, default=10)
    ap.add_argument("--adapted-layers", type=int, default=4)
    ap.add_argument("--mode", choices=["selection", "logit"], default="selection")
    ap.add_argument("--scope", choices=["scored", "all"], default="scored")
    ap.add_argument("--n-search", type=int, default=64)
    ap.add_argument("--n-val", type=int, default=256)
    ap.add_argument("--time-budget-min", type=float, default=45.0)
    ap.add_argument("--max-batch-tokens", type=int, default=32768)
    ap.add_argument("--model-id", default=OLMOE_ID)
    ap.add_argument("--tiny", action="store_true")
    args = ap.parse_args()
    t_start = time.perf_counter()

    def log(msg: str) -> None:
        print(f"[seed {args.seed}] {msg}", flush=True)

    if args.tiny:
        model, encode, data = tiny_olmoe(), byte_encode, tiny_data()
        args.n_search, args.n_val = min(args.n_search, 4), min(args.n_val, 4)
        args.adapted_layers = min(args.adapted_layers, 2)
    else:
        from evorouter.tasks.arc import load_arc

        model, tokenizer = load_model(args.model_id, device="cuda" if torch.cuda.is_available() else "cpu")
        encode, data = hf_encoder(tokenizer), load_arc("ARC-Challenge")

    config_path = args.run_dir / "config.json"
    if config_path.exists():
        saved = json.loads(config_path.read_text())
        cfg = SearchConfig(**saved["search"])
        layers, margins = tuple(saved["layers"]), tuple(saved["margins"])
        n_search, n_val = saved["n_search"], saved["n_val"]
        log(f"resuming {args.run_dir}")
    else:
        cfg = SearchConfig(
            seed=args.seed, optimizer=args.optimizer, popsize=args.popsize, sigma0=args.sigma0,
            max_generations=args.generations, val_every=args.val_every, mode=args.mode, scope=args.scope,
        )  # fmt: skip
        n_search, n_val = args.n_search, args.n_val
        layers = last_layers(model, args.adapted_layers)
        search, _ = search_val_split(data["train"], cfg.seed, n_search, n_val)
        reqs = Evaluator(model, search, encode).requests
        stats = topk_margins(
            router_logits_at_scored_positions(model, reqs, layers), model.config.num_experts_per_tok
        )
        margins = tuple(stats[layer]["median"] for layer in layers)
        args.run_dir.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps({
            "search": asdict(cfg), "layers": list(layers), "margins": list(margins),
            "n_search": n_search, "n_val": n_val, "model_id": args.model_id, "env": env_info(),
        }, indent=2))  # fmt: skip
        log(f"new run {args.run_dir}: layers={layers} margins={[round(m, 4) for m in margins]}")

    search, val = search_val_split(data["train"], cfg.seed, n_search, n_val)
    genome = BiasGenome(layers, model.config.num_experts, scale=margins)
    kw = dict(mode=cfg.mode, scope=cfg.scope, max_batch_tokens=args.max_batch_tokens)
    run = SearchRun(
        args.run_dir,
        cfg,
        search_eval=Evaluator(model, search, encode, genome, **kw),
        val_eval=Evaluator(model, val, encode, genome, **kw),
        test_eval=Evaluator(model, data["test"], encode, genome, **kw),
    )
    budget = args.time_budget_min * 60 - (time.perf_counter() - t_start)
    done = run.run(budget, log=log)
    log("finished" if done else "paused (resubmit the same command to continue)")


if __name__ == "__main__":
    main()
