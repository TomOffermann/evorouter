"""Diagnostic 1: single-expert sensitivity scan.

For every expert i in every adapted layer l, force it into (beta = +F) or out of (beta = -F) the
top-k at in-scope positions -- one intervention at a time -- and measure the change in fitness and
acc_norm on the search set and on the validation set, relative to the base model.

Questions it answers:
  * Does ANY single routing change move validation, and by how much?
  * Is the effect concentrated in a few experts (sparse genome) or diffuse (dense biases)?
  * Does the search-set effect predict the validation effect (Spearman over all moves)?

Sharded over GPUs; the last step merges shards into summary.json:
    python scripts/scan_experts.py --out-dir D --shard 0 --num-shards 4   # one per GPU
    python scripts/scan_experts.py --out-dir D --merge
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from evorouter.context import load_context, shard
from evorouter.diagnostics import expert_usage, router_logits_at_scored_positions
from evorouter.evaluate import Evaluator
from evorouter.genome import BiasGenome
from evorouter.runinfo import env_info
from evorouter.scoring import mc_scores, option_scores
from evorouter.stats import quantiles, spearman


def evaluate_all(ev: Evaluator, thetas: np.ndarray, chunk: int) -> np.ndarray:
    return np.concatenate([ev.loglik(thetas[i : i + chunk]) for i in range(0, len(thetas), chunk)])


def run_shard(args: argparse.Namespace) -> None:
    t0 = time.perf_counter()
    ctx = load_context(seed=args.seed, n_search=args.n_search, n_val=args.n_val,
                       adapted_layers=args.adapted_layers, tiny=args.tiny)  # fmt: skip
    num_e = ctx.num_experts
    genome = BiasGenome(ctx.layers, num_e)  # absolute logits: +-F forces an expert in / out
    items = [(j, e, d) for j in range(len(ctx.layers)) for e in range(num_e) for d in (1, -1)]
    mine = shard(items, args.shard, args.num_shards)

    thetas = np.zeros((len(mine) + 1, genome.dim))  # row 0 = base model
    for k, (j, e, d) in enumerate(mine, start=1):
        thetas[k, j * num_e + e] = d * args.force

    kw = dict(scope=args.scope, max_batch_tokens=args.max_batch_tokens)
    sets = {"search": Evaluator(ctx.model, ctx.search, ctx.encode, genome, **kw),
            "val": Evaluator(ctx.model, ctx.val, ctx.encode, genome, **kw)}  # fmt: skip
    results = {}
    for name, ev in sets.items():
        ll = evaluate_all(ev, thetas, args.chunk)
        s = mc_scores(ll, ev.requests, ev.questions)
        _, norm = option_scores(ll, ev.requests, ev.questions)
        answer = norm.argmax(-1)
        results[name] = {
            "fitness": s.fitness, "acc_norm": s.acc_norm,
            "flips": (answer != answer[:1]).mean(-1), "n_questions": len(ev.questions),
        }  # fmt: skip

    usage = expert_usage(
        router_logits_at_scored_positions(ctx.model, sets["search"].requests, ctx.layers), ctx.top_k
    )
    rows = []
    for k, (j, e, d) in enumerate(mine, start=1):
        layer = ctx.layers[j]
        row = {"layer": layer, "expert": e, "direction": "in" if d > 0 else "out",
               "base_usage": float(usage[layer][e])}  # fmt: skip
        for name, r in results.items():
            row[name] = {
                "d_fitness": float(r["fitness"][k] - r["fitness"][0]),
                "d_acc_norm": float(r["acc_norm"][k] - r["acc_norm"][0]),
                "flips": float(r["flips"][k]),
            }
        rows.append(row)

    out = {
        "args": {k: str(v) for k, v in vars(args).items()},
        "env": env_info(),
        "base": {name: {"fitness": float(r["fitness"][0]), "acc_norm": float(r["acc_norm"][0]),
                        "n_questions": r["n_questions"]} for name, r in results.items()},
        "rows": rows,
        "seconds": round(time.perf_counter() - t0, 1),
    }  # fmt: skip
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"shard{args.shard}.json").write_text(json.dumps(out, indent=1))
    print(f"shard {args.shard}: {len(rows)} moves in {out['seconds']} s", flush=True)


def merge(args: argparse.Namespace) -> None:
    shards = [json.loads(p.read_text()) for p in sorted(args.out_dir.glob("shard*.json"))]
    rows = [r for s in shards for r in s["rows"]]
    base = shards[0]["base"]
    n_val = base["val"]["n_questions"]
    s_fit = np.array([r["search"]["d_fitness"] for r in rows])
    v_fit = np.array([r["val"]["d_fitness"] for r in rows])
    v_acc = np.array([r["val"]["d_acc_norm"] for r in rows])

    def brief(r: dict) -> dict:
        return {
            "move": f"L{r['layer']} E{r['expert']} {r['direction']}", "usage": round(r["base_usage"], 3),
            "search_d_fit": round(r["search"]["d_fitness"], 5), "val_d_fit": round(r["val"]["d_fitness"], 5),
            "val_d_acc_norm_questions": round(r["val"]["d_acc_norm"] * n_val),
            "val_flips": round(r["val"]["flips"], 3),
        }  # fmt: skip

    layers = sorted({r["layer"] for r in rows})
    summary = {
        "n_moves": len(rows),
        "base": base,
        "spearman_search_vs_val_d_fitness": spearman(s_fit, v_fit),
        "val_d_fitness": quantiles(v_fit),
        "val_d_acc_norm_questions": quantiles(v_acc * n_val),
        "moves_improving_val_acc_norm_by_3plus_questions": int((v_acc * n_val >= 3).sum()),
        "moves_hurting_val_acc_norm_by_3plus_questions": int((v_acc * n_val <= -3).sum()),
        "mean_abs_val_d_fitness_by_layer": {
            str(layer): float(np.mean([abs(r["val"]["d_fitness"]) for r in rows if r["layer"] == layer]))
            for layer in layers
        },
        "mean_abs_val_d_fitness_by_direction": {
            d: float(np.mean([abs(r["val"]["d_fitness"]) for r in rows if r["direction"] == d]))
            for d in ("in", "out")
        },
        "top10_by_val": [brief(rows[i]) for i in np.argsort(-v_fit)[:10]],
        "top10_by_search": [brief(rows[i]) for i in np.argsort(-s_fit)[:10]],
        "bottom5_by_val": [brief(rows[i]) for i in np.argsort(v_fit)[:5]],
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--force", type=float, default=50.0, help="|bias| in logits that forces an expert")
    ap.add_argument("--scope", choices=["scored", "all"], default="scored")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-search", type=int, default=64)
    ap.add_argument("--n-val", type=int, default=256)
    ap.add_argument("--adapted-layers", type=int, default=4)
    ap.add_argument("--max-batch-tokens", type=int, default=32768)
    ap.add_argument("--chunk", type=int, default=64, help="genomes per evaluator call")
    ap.add_argument("--tiny", action="store_true")
    args = ap.parse_args()
    merge(args) if args.merge else run_shard(args)


if __name__ == "__main__":
    main()
