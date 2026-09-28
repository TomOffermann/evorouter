"""Model and tokenizer loading, plus a tiny random OLMoE for CPU tests."""

from __future__ import annotations

import torch
from torch import nn

from evorouter.scoring import Encode

OLMOE_ID = "allenai/OLMoE-1B-7B-0924"


def load_model(model_id: str = OLMOE_ID, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
    """Frozen model in eval mode + its tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device).eval()
    model.requires_grad_(False)
    return model, tokenizer


def hf_encoder(tokenizer) -> Encode:
    """lm-eval-harness convention for decoder-only models: no special tokens added."""
    return lambda text: tokenizer(text, add_special_tokens=False).input_ids


def byte_encode(text: str) -> list[int]:
    """Offline tokenizer for tests and --tiny runs: one id per UTF-8 byte."""
    return list(text.encode())


def tiny_olmoe(
    num_layers: int = 3, num_experts: int = 8, top_k: int = 2, vocab_size: int = 256, seed: int = 0
) -> nn.Module:
    """Random-weight OLMoE small enough for CPU unit tests (float32, eval mode)."""
    from transformers import OlmoeConfig, OlmoeForCausalLM

    cfg = OlmoeConfig(
        vocab_size=vocab_size,
        hidden_size=32,
        intermediate_size=16,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_experts=num_experts,
        num_experts_per_tok=top_k,
        norm_topk_prob=False,
        pad_token_id=0,
    )
    torch.manual_seed(seed)
    model = OlmoeForCausalLM(cfg).eval()
    model.requires_grad_(False)
    return model
