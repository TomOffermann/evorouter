"""Resumable black-box search over selection-bias genomes.

One :class:`SearchRun` = one seed: fixed search and validation sets, an optimizer, and a run
directory holding everything needed to resume after a job ends (every Slurm job is <= 1 h):

    run_dir/config.json      the SearchConfig + margins (fixed at creation)
    run_dir/state.pkl        optimizer state, generation counter, best-validation record
    run_dir/metrics.jsonl    one line per generation (search set)
    run_dir/val.jsonl        one line per validation checkpoint
    run_dir/final.json       test-set result of the selected genome (written once, at the end)

Optimizers (all maximize the search-set fitness; only ranks are used):
    cma      full-covariance CMA-ES (pycma)
    sep-cma  diagonal CMA-ES (pycma option CMA_diagonal)
    random   i.i.d. N(0, sigma0^2 I) samples, incumbent = best seen (noise control)
"""

from __future__ import annotations

import json
import pickle
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from evorouter.evaluate import Evaluator
from evorouter.scoring import MCScores

Optimizer = Literal["cma", "sep-cma", "random"]


@dataclass(frozen=True)
class SearchConfig:
    seed: int = 0
    optimizer: Optimizer = "cma"
    popsize: int = 64
    sigma0: float = 1.0  # in margin units: beta_{l,i} = Delta_l * theta_{l,i}
    max_generations: int = 200
    val_every: int = 10
    mode: str = "selection"
    scope: str = "scored"


class RandomSearch:
    """ask/tell interface matching pycma; the 'mean' is the best candidate seen so far."""

    def __init__(self, dim: int, sigma0: float, popsize: int, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.dim, self.sigma, self.popsize = dim, sigma0, popsize
        self.mean = np.zeros(dim)
        self.best_f = np.inf  # minimization convention, as pycma

    def ask(self) -> list[np.ndarray]:
        return list(self.rng.normal(0.0, self.sigma, size=(self.popsize, self.dim)))

    def tell(self, xs: list[np.ndarray], fs: list[float]) -> None:
        i = int(np.argmin(fs))
        if fs[i] < self.best_f:
            self.best_f, self.mean = fs[i], np.asarray(xs[i])


def make_optimizer(cfg: SearchConfig, dim: int):
    if cfg.optimizer == "random":
        return RandomSearch(dim, cfg.sigma0, cfg.popsize, cfg.seed)
    import cma

    opts = {"popsize": cfg.popsize, "seed": cfg.seed + 1, "verbose": -9}
    if cfg.optimizer == "sep-cma":
        opts["CMA_diagonal"] = True
    elif cfg.optimizer != "cma":
        raise ValueError(f"unknown optimizer {cfg.optimizer!r}")
    return cma.CMAEvolutionStrategy(np.zeros(dim), cfg.sigma0, opts)


def optimizer_mean(opt) -> np.ndarray:
    """Current point estimate: the CMA-ES distribution mean, or random search's incumbent."""
    return np.asarray(opt.mean, dtype=float)


def optimizer_sigma(opt) -> float:
    return float(opt.sigma)


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


class SearchRun:
    def __init__(
        self,
        run_dir: Path,
        cfg: SearchConfig,
        search_eval: Evaluator,
        val_eval: Evaluator,
        test_eval: Evaluator | None = None,
    ) -> None:
        if search_eval.genome is None:
            raise ValueError("search evaluator needs a genome")
        self.dir, self.cfg = Path(run_dir), cfg
        self.search_eval, self.val_eval, self.test_eval = search_eval, val_eval, test_eval
        self.dim = search_eval.genome.dim
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.dir / "state.pkl"
        if self.state_path.exists():
            with self.state_path.open("rb") as f:
                self.state = pickle.load(f)
        else:
            self.state = {
                "generation": 0,
                "evaluations": 0,
                "optimizer": make_optimizer(cfg, self.dim),
                "best_val": None,  # {"generation", "fitness", "acc_norm", "theta"}
                "np_random": None,
            }

    # --- persistence -------------------------------------------------------------------------
    def save(self) -> None:
        self.state["np_random"] = np.random.get_state()  # pycma draws from the global RNG
        tmp = self.state_path.with_suffix(".tmp")
        with tmp.open("wb") as f:
            pickle.dump(self.state, f)
        tmp.replace(self.state_path)  # atomic: a killed job never leaves a half-written state

    @property
    def done(self) -> bool:
        return self.state["generation"] >= self.cfg.max_generations

    # --- one step ----------------------------------------------------------------------------
    def _validate(self) -> dict[str, Any]:
        theta = optimizer_mean(self.state["optimizer"])
        s = self.val_eval.evaluate(theta[None])
        row = {"generation": self.state["generation"], **s.member(0)}
        best = self.state["best_val"]
        if best is None or row["fitness"] > best["fitness"]:
            self.state["best_val"] = {**row, "theta": theta.copy()}
        _append_jsonl(self.dir / "val.jsonl", row)
        return row

    def step(self) -> dict[str, Any]:
        opt = self.state["optimizer"]
        t0 = time.perf_counter()
        xs = opt.ask()
        scores: MCScores = self.search_eval.evaluate(np.stack(xs))
        opt.tell(xs, list(-scores.fitness))  # optimizers minimize
        self.state["generation"] += 1
        self.state["evaluations"] += len(xs)
        best = int(np.argmax(scores.fitness))
        row = {
            "generation": self.state["generation"],
            "evaluations": self.state["evaluations"],
            "fitness_best": float(scores.fitness[best]),
            "fitness_mean": float(scores.fitness.mean()),
            "fitness_median": float(np.median(scores.fitness)),
            "acc_norm_best": float(scores.acc_norm[best]),
            "n_distinct_fitness": int(len(np.unique(np.round(scores.fitness, 10)))),
            "sigma": optimizer_sigma(opt),
            "mean_norm": float(np.linalg.norm(optimizer_mean(opt))),
            "seconds": round(time.perf_counter() - t0, 2),
        }
        _append_jsonl(self.dir / "metrics.jsonl", row)
        if self.state["generation"] % self.cfg.val_every == 0 or self.done:
            row["val"] = self._validate()
        return row

    # --- driver ------------------------------------------------------------------------------
    def run(self, time_budget_s: float, log=print) -> bool:
        """Run until done or out of time; saves after every generation. Returns True when done."""
        if self.state["np_random"] is not None:
            np.random.set_state(self.state["np_random"])
        if self.state["generation"] == 0 and self.state["best_val"] is None:
            self._validate()  # generation-0 reference: the base model (theta = 0)
            self.save()
        start = time.perf_counter()
        last = 0.0
        while not self.done:
            if time.perf_counter() - start + last > time_budget_s:
                log(f"time budget reached at generation {self.state['generation']}; state saved")
                return False
            t = time.perf_counter()
            row = self.step()
            self.save()
            last = time.perf_counter() - t
            log(json.dumps(row))
        self.finish(log)
        return True

    def finish(self, log=print) -> dict[str, Any] | None:
        final_path = self.dir / "final.json"
        if final_path.exists() or self.test_eval is None:
            return None
        best = self.state["best_val"]
        theta = best["theta"]
        test = self.test_eval.evaluate(np.stack([np.zeros(self.dim), theta]))
        result = {
            "config": asdict(self.cfg),
            "selected_generation": best["generation"],
            "val": {k: v for k, v in best.items() if k != "theta"},
            "test_base": test.member(0),
            "test_selected": test.member(1),
            "delta_acc_norm": float(test.acc_norm[1] - test.acc_norm[0]),
            "evaluations": self.state["evaluations"],
            "theta": theta.tolist(),
        }
        final_path.write_text(json.dumps(result, indent=2))
        log(f"final: {json.dumps({k: v for k, v in result.items() if k != 'theta'})}")
        return result
