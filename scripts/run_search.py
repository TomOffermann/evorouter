"""Search: evolve a selection-only bias on the last MoE layers of OLMoE for ARC-Challenge.

Data protocol (see evorouter.search): a fresh mini-batch of the ARC-C *train* split every
generation, model selection on the official *validation* split, one final evaluation on *test*.

Resumable: re-running with the same --run-dir continues from the saved state (settings, layers and
margins are read from the run directory; CLI search settings are ignored on resume).

    python scripts/run_search.py --run-dir runs/cma-v1/seed0 --seed 0 --optimizer cma
    python scripts/run_search.py --tiny --run-dir /tmp/tiny --generations 3 --popsize 4 --batch-size 4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from evorouter.context import load_context
from evorouter.evaluate import Evaluator
from evorouter.genome import BiasGenome
from evorouter.runinfo import env_info
from evorouter.search import SearchConfig, SearchRun

PROTOCOL = "arc-train-minibatch/validation/test-v1"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--optimizer", choices=["cma", "sep-cma", "random"], default="cma")
    ap.add_argument("--popsize", type=int, default=64)
    ap.add_argument("--sigma0", type=float, default=1.0)
    ap.add_argument("--generations", type=int, default=200)
    ap.add_argument("--val-every", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=128, help="train questions per generation; 0 = all")
    ap.add_argument("--adapted-layers", type=int, default=4)
    ap.add_argument("--mode", choices=["selection", "logit"], default="selection")
    ap.add_argument("--scope", choices=["scored", "all"], default="scored")
    ap.add_argument("--margin-questions", type=int, default=128, help="train sample for router margins")
    ap.add_argument("--time-budget-min", type=float, default=45.0)
    ap.add_argument("--max-batch-tokens", type=int, default=32768)
    ap.add_argument("--tiny", action="store_true")
    args = ap.parse_args()
    t_start = time.perf_counter()

    def log(msg: str) -> None:
        print(f"[seed {args.seed}] {msg}", flush=True)

    config_path = args.run_dir / "config.json"
    saved = json.loads(config_path.read_text()) if config_path.exists() else None
    if saved is not None and saved.get("protocol") != PROTOCOL:
        sys.exit(f"{args.run_dir} was created with an older data protocol; use a new --run-dir / NAME")

    seed = saved["search"]["seed"] if saved else args.seed
    adapted = len(saved["layers"]) if saved else args.adapted_layers
    ctx = load_context(seed=seed, n_search=args.margin_questions, adapted_layers=adapted, tiny=args.tiny)

    if saved:
        cfg = SearchConfig(**saved["search"])
        layers, margins = tuple(saved["layers"]), tuple(saved["margins"])
        log(f"resuming {args.run_dir} at its saved settings")
    else:
        batch_size = min(args.batch_size, len(ctx.data["train"]))
        cfg = SearchConfig(
            seed=args.seed, optimizer=args.optimizer, popsize=args.popsize, sigma0=args.sigma0,
            max_generations=args.generations, val_every=args.val_every, batch_size=batch_size,
            mode=args.mode, scope=args.scope,
        )  # fmt: skip
        layers, margins = ctx.layers, ctx.margins
        args.run_dir.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps({
            "protocol": PROTOCOL, "search": asdict(cfg), "layers": list(layers), "margins": list(margins),
            "splits": {k: len(v) for k, v in ctx.data.items()}, "env": env_info(),
        }, indent=2))  # fmt: skip
        log(f"new run {args.run_dir}: layers={layers} margins={[round(m, 4) for m in margins]}")

    genome = BiasGenome(layers, ctx.num_experts, scale=margins)
    kw = dict(mode=cfg.mode, scope=cfg.scope, max_batch_tokens=args.max_batch_tokens)
    run = SearchRun(
        args.run_dir,
        cfg,
        train_eval=Evaluator(ctx.model, ctx.data["train"], ctx.encode, genome, **kw),
        val_eval=Evaluator(ctx.model, ctx.data["validation"], ctx.encode, genome, **kw),
        test_eval=Evaluator(ctx.model, ctx.data["test"], ctx.encode, genome, **kw),
    )
    budget = args.time_budget_min * 60 - (time.perf_counter() - t_start)
    done = run.run(budget, log=log)
    log("finished" if done else "paused (resubmit the same command to continue)")


if __name__ == "__main__":
    main()
