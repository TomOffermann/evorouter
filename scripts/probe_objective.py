"""Diagnostic 2: how much can routing change the answer scores at all, and how should tau be set?

For each scope in {scored, all} and step size sigma (margin units, exactly as in the search), sample
K random genomes and evaluate them on the search and validation sets. Temperatures are applied post
hoc from the same log-likelihoods (no extra forward passes): the given taus plus tau*, fitted on the
base model's search-set scores.

Per (scope, sigma) it reports
  * selection change rate per layer (share of in-scope positions whose top-k set changes),
  * mean |dLL| per continuation token and the share of questions whose acc_norm answer flips,
  * validation acc_norm of the genomes vs base,
  * per tau: fitness spread across genomes, and the Spearman correlation between search and
    validation fitness across genomes (does the search objective rank genomes the same way on
    held-out questions?), plus best-of-K transfer (pick the best genome on search, report val).

    python scripts/probe_objective.py --out-dir D --shard 0 --num-shards 4   # one per GPU
    python scripts/probe_objective.py --out-dir D --merge
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from evorouter.context import load_context, shard
from evorouter.diagnostics import selection_change_rate
from evorouter.evaluate import Evaluator
from evorouter.genome import BiasGenome
from evorouter.runinfo import env_info
from evorouter.scoring import fit_temperature, mean_log_q, option_scores
from evorouter.stats import spearman


def run_shard(args: argparse.Namespace) -> None:
    t0 = time.perf_counter()
    ctx = load_context(seed=args.seed, n_search=args.n_search, n_val=args.n_val,
                       adapted_layers=args.adapted_layers, tiny=args.tiny)  # fmt: skip
    genome = BiasGenome(ctx.layers, ctx.num_experts, scale=ctx.margins)
    configs = [(scope, sigma) for scope in args.scopes for sigma in args.sigmas]
    mine = shard(configs, args.shard, args.num_shards)

    evs = {
        (scope, name): Evaluator(ctx.model, qs, ctx.encode, genome, scope=scope,
                                 max_batch_tokens=args.max_batch_tokens)
        for scope in args.scopes for name, qs in (("search", ctx.search), ("val", ctx.val))
    }  # fmt: skip
    base, answers, n_cont = {}, {}, {}
    for name in ("search", "val"):
        ev = evs[(args.scopes[0], name)]
        base[name] = ev.loglik(None)  # scope does not matter for the base model
        answers[name] = np.array([q.answer for q in ev.questions])
        n_cont[name] = np.array([r.n_cont for r in ev.requests])
    ev_s = evs[(args.scopes[0], "search")]
    tau_star = fit_temperature(base["search"], ev_s.requests, ev_s.questions)
    taus = sorted({*args.taus, tau_star}, reverse=True)

    results = []
    for scope, sigma in mine:
        rng = np.random.default_rng([args.seed, int(round(sigma * 1000)), args.scopes.index(scope)])
        thetas = rng.normal(0.0, sigma, size=(args.k, genome.dim))
        row: dict = {"scope": scope, "sigma": sigma}
        norms = {}
        for name in ("search", "val"):
            ev = evs[(scope, name)]
            ll = ev.loglik(thetas)
            _, norm = option_scores(ll, ev.requests, ev.questions)
            _, norm_base = option_scores(base[name], ev.requests, ev.questions)
            acc_norm = (norm.argmax(-1) == answers[name]).mean(-1)
            norms[name] = (norm, norm_base)
            row[name] = {
                "mean_abs_dll_per_token": float(np.mean(np.abs(ll - base[name]) / n_cont[name])),
                "answer_flip_rate": float((norm.argmax(-1) != norm_base.argmax(-1)).mean()),
                "acc_norm_base": float((norm_base.argmax(-1) == answers[name]).mean()),
                "acc_norm_mean": float(acc_norm.mean()),
                "acc_norm_max": float(acc_norm.max()),
            }
        row["tau"] = {}
        for tau in taus:
            f = {n: mean_log_q(norms[n][0], answers[n], tau) for n in ("search", "val")}
            fb = {n: float(mean_log_q(norms[n][1], answers[n], tau)[0]) for n in ("search", "val")}
            best = int(np.argmax(f["search"]))
            row["tau"][f"{tau:.4g}"] = {
                "search_fitness_std": float(f["search"].std()),
                "search_d_fitness_mean": float(f["search"].mean() - fb["search"]),
                "spearman_search_vs_val": spearman(f["search"], f["val"]),
                "best_of_k_val_d_fitness": float(f["val"][best] - fb["val"]),
            }
        reqs = ev_s.requests[: sum(len(q.choices) for q in ctx.search[: args.change_questions])]
        rates = [
            selection_change_rate(ctx.model, reqs, genome.biases(thetas[i]), scope=scope)
            for i in range(min(4, args.k))
        ]
        row["selection_change_rate"] = {
            str(layer): float(np.mean([r[layer] for r in rates])) for layer in ctx.layers
        }
        results.append(row)
        print(f"shard {args.shard}: scope={scope} sigma={sigma} done", flush=True)

    out = {
        "args": {k: str(v) for k, v in vars(args).items()},
        "env": env_info(),
        "margins": list(ctx.margins),
        "tau_star": tau_star,
        "results": results,
        "seconds": round(time.perf_counter() - t0, 1),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"shard{args.shard}.json").write_text(json.dumps(out, indent=1))


def merge(args: argparse.Namespace) -> None:
    shards = [json.loads(p.read_text()) for p in sorted(args.out_dir.glob("shard*.json"))]
    rows = sorted((r for s in shards for r in s["results"]), key=lambda r: (r["scope"], r["sigma"]))
    summary = {"tau_star": shards[0]["tau_star"], "margins": shards[0]["margins"], "results": rows}
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"tau* (fitted on base search scores) = {summary['tau_star']:.4g}")
    taus = list(rows[0]["tau"])
    head = "scope   sigma  sel_change  |dLL|/tok  flips_val  acc_val(base/mean/max)  " + "  ".join(
        f"rho@{t}" for t in taus
    )
    print(head)
    for r in rows:
        sel = np.mean(list(r["selection_change_rate"].values()))
        v = r["val"]
        rhos = "  ".join(f"{r['tau'][t]['spearman_search_vs_val']:+.2f}".rjust(len(f"rho@{t}")) for t in taus)
        print(
            f"{r['scope']:7} {r['sigma']:5.1f}  {sel:10.3f}  {v['mean_abs_dll_per_token']:9.4f}  "
            f"{v['answer_flip_rate']:9.3f}  {v['acc_norm_base']:.3f}/{v['acc_norm_mean']:.3f}/"
            f"{v['acc_norm_max']:.3f}        {rhos}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--scopes", nargs="+", default=["scored", "all"], choices=["scored", "all"])
    ap.add_argument("--sigmas", nargs="+", type=float, default=[0.5, 1.0, 3.0, 10.0])
    ap.add_argument("--k", type=int, default=32, help="random genomes per (scope, sigma)")
    ap.add_argument("--taus", nargs="+", type=float, default=[1.0, 0.3, 0.1, 0.03])
    ap.add_argument("--change-questions", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-search", type=int, default=128, help="train sample")
    ap.add_argument("--n-val", type=int, default=None, help="first N validation questions")
    ap.add_argument("--adapted-layers", type=int, default=4)
    ap.add_argument("--max-batch-tokens", type=int, default=32768)
    ap.add_argument("--tiny", action="store_true")
    args = ap.parse_args()
    merge(args) if args.merge else run_shard(args)


if __name__ == "__main__":
    main()
