# evorouter

Gradient-free (CMA-ES) search over the discrete expert selection of frozen Mixture-of-Experts
language models: task-level "routing personas" found by evolving selection-only router biases,
with cached, exact, hard-routed fitness evaluation.

Status: environment bring-up. v0 experiment = OLMoE-1B-7B on ARC-Challenge (see design doc).

## Layout (growing)

```
cluster/clariden/   container + venv setup, Slurm scripts, cluster README
scripts/            entry points (smoke_test.py)
```

## Quick checks

```bash
# local CPU, no Hub access needed: random tiny OLMoE, byte-level tokenizer fallback
python scripts/smoke_test.py --tiny --device cpu --skip-data
```

Cluster: see `cluster/clariden/README.md`.
