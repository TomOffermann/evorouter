"""Clariden smoke test for evorouter.

Each check reports PASS / FAIL / SKIP; everything is also written as JSON to --out.

  env    python / torch / CUDA, CPU architecture, visible GPUs, package versions
  gpus   bf16 matmul on every visible GPU (sanity + rough TFLOP/s)
  model  OLMoE-1B-7B loads in bf16; router config and module layout are as we assume
  score  lm-eval-style option log-likelihoods on one ARC-like question
  hook   a forward hook on the last router captures logits; median top-k margin
  batch  forward throughput on a 64 x 64-token batch (for runtime planning)
  data   ARC-Challenge downloads with the expected split sizes

On the cluster:   python scripts/smoke_test.py --out smoke.json
Locally (CPU):    python scripts/smoke_test.py --tiny --device cpu --skip-data
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import torch

MODEL_ID = "allenai/OLMoE-1B-7B-0924"
EXPECTED_CONFIG = {
    "num_experts": 64,
    "num_experts_per_tok": 8,
    "num_hidden_layers": 16,
    "hidden_size": 2048,
    "norm_topk_prob": False,
}
ARC_SIZES = {"train": 1119, "validation": 299, "test": 1172}
QUESTION = "Which of these is a renewable source of energy?"
OPTIONS = ["coal", "natural gas", "wind", "oil"]  # correct: "wind"


class Smoke:
    def __init__(self) -> None:
        self.results: dict[str, dict[str, Any]] = {}

    def run(self, name: str, fn: Callable[[], dict[str, Any]], skip: str | None = None) -> None:
        if skip:
            self.results[name] = {"status": "SKIP", "reason": skip}
            print(f"[SKIP] {name}: {skip}", flush=True)
            return
        t0 = time.perf_counter()
        try:
            detail = fn()
            status = "PASS"
        except Exception as exc:  # report and continue with the other checks
            detail = {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
            status = "FAIL"
        detail["seconds"] = round(time.perf_counter() - t0, 2)
        self.results[name] = {"status": status, **detail}
        shown = {k: v for k, v in detail.items() if k != "traceback"}
        print(f"[{status}] {name}: {json.dumps(shown, default=str)}", flush=True)
        if status == "FAIL":
            print(detail["traceback"], flush=True)

    def ok(self, name: str) -> bool:
        return self.results.get(name, {}).get("status") == "PASS"


def _version(pkg: str) -> str | None:
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return None


def check_env(device: str) -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "machine": platform.machine(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "packages": {p: _version(p) for p in ("transformers", "datasets", "accelerate", "cma")},
    }
    if torch.cuda.is_available():
        info["gpus"] = [
            {
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "mem_gb": round(torch.cuda.get_device_properties(i).total_memory / 2**30, 1),
            }
            for i in range(torch.cuda.device_count())
        ]
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda requested but CUDA is not available")
    missing = [p for p, v in info["packages"].items() if v is None]
    if missing:
        raise RuntimeError(f"missing packages: {missing}")
    return info


def check_gpus(n: int = 8192, reps: int = 10) -> dict[str, Any]:
    out = {}
    for i in range(torch.cuda.device_count()):
        dev = f"cuda:{i}"
        a = torch.randn(n, n, device=dev, dtype=torch.bfloat16)
        b = torch.randn(n, n, device=dev, dtype=torch.bfloat16)
        (a @ b).sum().item()  # warm-up
        torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        for _ in range(reps):
            c = a @ b
        torch.cuda.synchronize(dev)
        dt = time.perf_counter() - t0
        if not torch.isfinite(c).all():
            raise RuntimeError(f"non-finite matmul result on {dev}")
        out[dev] = round(2 * n**3 * reps / dt / 1e12, 1)
    return {"bf16_matmul_tflops": out}


class ByteTokenizer:
    """Offline stand-in (one id per UTF-8 byte) so --tiny runs without Hub access."""

    def __len__(self) -> int:
        return 256

    def __call__(self, text: str, return_tensors: str | None = None):
        ids = list(text.encode())
        return argparse.Namespace(input_ids=torch.tensor([ids]) if return_tensors == "pt" else ids)


def load_model(model_id: str, device: str, tiny: bool):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    except Exception:
        if not tiny:
            raise
        tokenizer = ByteTokenizer()
    if tiny:
        from transformers import OlmoeConfig, OlmoeForCausalLM

        cfg = OlmoeConfig(
            vocab_size=len(tokenizer), hidden_size=64, intermediate_size=32, num_hidden_layers=4,
            num_attention_heads=4, num_key_value_heads=4, num_experts=8, num_experts_per_tok=2,
            norm_topk_prob=False,
        )
        torch.manual_seed(0)
        model = OlmoeForCausalLM(cfg)
    else:
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    return model.to(device).eval(), tokenizer


def describe_model(model, tiny: bool) -> dict[str, Any]:
    cfg = model.config
    got = {k: getattr(cfg, k, None) for k in EXPECTED_CONFIG}
    layer = model.model.layers[-1]
    moe = layer.mlp
    gate = getattr(moe, "gate", None)
    info = {
        "config": got,
        "params_b": round(sum(p.numel() for p in model.parameters()) / 1e9, 3),
        "moe_block": type(moe).__name__,
        "router": type(gate).__name__ if gate is not None else None,
        "router_weight_shape": list(gate.weight.shape) if gate is not None else None,
        "moe_attrs": {a: getattr(moe, a, None) for a in ("top_k", "num_experts", "norm_topk_prob")},
    }
    if gate is None:
        raise RuntimeError("layer.mlp.gate not found: router hook target differs from assumption")
    if not tiny:
        bad = {k: (got[k], v) for k, v in EXPECTED_CONFIG.items() if got[k] != v}
        if bad:
            raise RuntimeError(f"config differs from expectation (got, expected): {bad}")
    if torch.cuda.is_available():
        info["gpu_mem_allocated_gb"] = round(torch.cuda.memory_allocated() / 2**30, 2)
    return info


@torch.no_grad()
def option_scores(model, tokenizer, device: str) -> dict[str, Any]:
    ctx = f"Question: {QUESTION}\nAnswer:"
    n_ctx = len(tokenizer(ctx).input_ids)
    rows = []
    for opt in OPTIONS:
        cont = " " + opt
        ids = tokenizer(ctx + cont, return_tensors="pt").input_ids.to(device)
        logp = torch.log_softmax(model(ids).logits[0, :-1].float(), dim=-1)
        targets = ids[0, n_ctx:]
        ll = logp[n_ctx - 1 :].gather(-1, targets[:, None]).sum().item()
        rows.append({"option": opt, "tokens": int(targets.numel()), "ll": round(ll, 3),
                     "ll_per_byte": round(ll / len(cont.encode()), 4)})
    if not all(abs(r["ll"]) < float("inf") for r in rows):
        raise RuntimeError("non-finite log-likelihood")
    return {
        "options": rows,
        "argmax_acc": max(rows, key=lambda r: r["ll"])["option"],
        "argmax_acc_norm": max(rows, key=lambda r: r["ll_per_byte"])["option"],
        "expected": "wind",
    }


@torch.no_grad()
def router_hook(model, tokenizer, device: str) -> dict[str, Any]:
    captured: list[torch.Tensor] = []
    gate = model.model.layers[-1].mlp.gate
    handle = gate.register_forward_hook(lambda _m, _i, out: captured.append(out.detach().float()))
    try:
        ids = tokenizer(f"Question: {QUESTION}\nAnswer: wind", return_tensors="pt").input_ids.to(device)
        model(ids)
    finally:
        handle.remove()
    if not captured:
        raise RuntimeError("router hook did not fire")
    logits = captured[0].reshape(-1, captured[0].shape[-1])  # [tokens, experts]
    k = model.config.num_experts_per_tok
    top = logits.topk(k + 1, dim=-1).values
    margin = (top[:, k - 1] - top[:, k]).median().item()
    return {"router_logits_shape": list(logits.shape), "median_topk_margin": round(margin, 4),
            "logit_std": round(logits.std().item(), 4)}


@torch.no_grad()
def throughput(model, device: str, batch: int = 64, seq: int = 64, reps: int = 3) -> dict[str, Any]:
    ids = torch.randint(0, model.config.vocab_size, (batch, seq), device=device)
    model(ids)  # warm-up
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(reps):
        model(ids)
    if device == "cuda":
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / reps
    out = {"batch": batch, "seq": seq, "sec_per_forward": round(dt, 3),
           "tokens_per_sec": int(batch * seq / dt)}
    if device == "cuda":
        out["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    return out


def check_data() -> dict[str, Any]:
    from datasets import load_dataset

    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge")
    sizes = {split: len(ds[split]) for split in ds}
    if sizes != ARC_SIZES:
        raise RuntimeError(f"unexpected split sizes {sizes}, expected {ARC_SIZES}")
    ex = ds["test"][0]
    n_choices = sorted({len(r["choices"]["text"]) for r in ds["test"]})
    return {"sizes": sizes, "example_keys": sorted(ex.keys()), "choices_per_question": n_choices}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("smoke.json"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--tiny", action="store_true", help="random 4-layer OLMoE instead of the real weights")
    ap.add_argument("--skip-data", action="store_true")
    args = ap.parse_args()

    s = Smoke()
    s.run("env", lambda: check_env(args.device))
    s.run("gpus", check_gpus, skip=None if args.device == "cuda" else "no CUDA device")

    state: dict[str, Any] = {}

    def _load() -> dict[str, Any]:
        t0 = time.perf_counter()
        state["model"], state["tok"] = load_model(args.model_id, args.device, args.tiny)
        info = describe_model(state["model"], args.tiny)
        info["load_seconds"] = round(time.perf_counter() - t0, 1)
        return info

    s.run("model", _load)
    no_model = None if "model" in state else "model did not load"
    s.run("score", lambda: option_scores(state["model"], state["tok"], args.device), skip=no_model)
    s.run("hook", lambda: router_hook(state["model"], state["tok"], args.device), skip=no_model)
    s.run("batch", lambda: throughput(state["model"], args.device), skip=no_model)
    s.run("data", check_data, skip="--skip-data" if args.skip_data else None)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"args": vars(args), "results": s.results}, indent=2, default=str))
    failed = [n for n, r in s.results.items() if r["status"] == "FAIL"]
    print("\n" + "\n".join(f"  {r['status']:4}  {n}" for n, r in s.results.items()))
    print(f"\nwrote {args.out}" + (f" | FAILED: {failed}" if failed else " | all checks passed"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
