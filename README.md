# evorouter

Gradient-free (CMA-ES) search over the discrete expert selection of frozen Mixture-of-Experts
language models: task-level "routing personas" found by evolving selection-only router biases,
scored with exact, hard-routed fitness.

Status: milestone 1 done (base ARC-C acc_norm 48.8, matches lm-eval 48.9). Milestone 2: CMA-ES search.

Data protocol (v1): search on fresh mini-batches of the ARC-C **train** split (a new batch every
generation), select on the official **validation** split, report once on **test**. The v0 protocol
(64 fixed search questions) overfit and is retired; run directories record their protocol.

## Layout

```
src/evorouter/
  routing.py       selection-only router bias on OLMoE MoE blocks (the only HF-internals module)
  genome.py        flat genome theta <-> per-layer biases (layer-major, optional margin scaling)
  scoring.py       lm-eval-style cloze requests; acc / acc_norm / fitness from log-likelihoods
  evaluate.py      naive exact evaluator: a population of genomes in padded batches (ground truth)
  diagnostics.py   router statistics: top-k margins, expert usage/loads, selection change rate
  search.py        resumable search runs (CMA-ES, sep-CMA-ES, random); per-generation mini-batches
  context.py       shared script setup (model, data, split, layers, margins) and GPU sharding
  stats.py         rank statistics (Spearman) without scipy
  tasks/           MCQuestion, splits, ARC loader
  models.py        model/tokenizer loading, tiny random OLMoE for tests
  runinfo.py       provenance (git commit, versions, GPU) for result files
scripts/           entry points: smoke_test.py, eval_base.py, run_search.py,
                   scan_experts.py (single-expert scan), probe_objective.py (scope/sigma/tau probe)
tests/             CPU tests on a tiny random OLMoE (no network, seconds)
cluster/clariden/  container + venv setup and Slurm jobs
```

## Invariants (enforced by tests)

- A zero genome reproduces the HF forward bit-for-bit (`test_zero_bias_is_bit_identical_to_base`).
- `mode="selection"` changes which experts run, never their gate values.
- A bias restricted to some positions leaves all earlier positions unchanged (causality).
- Evaluating a population in one batch equals evaluating each genome alone.

## Development

```bash
pip install -e ".[dev]"
pytest                   # ~5 s on CPU
ruff check . && ruff format --check .
python scripts/eval_base.py --tiny --out /tmp/base.json   # offline end-to-end run
python scripts/run_search.py --tiny --run-dir /tmp/run --generations 3 --popsize 4
```

## Compute rules

- **Every Slurm job must finish within 1 hour.** Longer experiments are split into resumable
  jobs (checkpoint after every CMA-ES generation) and chained.
- One node = 4 GH200; run independent seeds on separate GPUs within one job.

Cluster instructions: `cluster/clariden/README.md`.
