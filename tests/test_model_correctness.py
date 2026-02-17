#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Application tests for PagedAttention v2 kernel mode correctness (AMD MI300X).

Tests that real model inference produces identical outputs with each kernel
mode. Uses greedy decoding (temperature=0) for deterministic results.

All 5 kernel modes must produce the exact same token sequence:
  1. baseline       - Original 2-kernel path
  2. fused          - Skip reduce for short seqs
  3. mfma4_all      - MFMA4 for all GQA ratios + fused
  4. mfma4_all_512  - MFMA4 all GQA + 512-token partition
  5. mfma4_all_1024 - MFMA4 all GQA + 1024-token partition

Each mode runs in a separate subprocess because the env vars that control
kernel behavior are cached in static C++ variables.
"""

import json
import os
import subprocess
import sys

import pytest

# ---------------------------------------------------------------------------
# Kernel mode env var configurations
# ---------------------------------------------------------------------------
KERNEL_MODES = {
    "baseline": {
        "VLLM_ROCM_FUSED_SHORT_SEQ": "0",
        "VLLM_ROCM_MFMA4_ALL": "0",
        "VLLM_ROCM_PARTITION_512": "0",
        "VLLM_ROCM_PARTITION_1024": "0",
    },
    "fused": {
        "VLLM_ROCM_FUSED_SHORT_SEQ": "1",
        "VLLM_ROCM_MFMA4_ALL": "0",
        "VLLM_ROCM_PARTITION_512": "0",
        "VLLM_ROCM_PARTITION_1024": "0",
    },
    "mfma4_all": {
        "VLLM_ROCM_FUSED_SHORT_SEQ": "1",
        "VLLM_ROCM_MFMA4_ALL": "1",
        "VLLM_ROCM_PARTITION_512": "0",
        "VLLM_ROCM_PARTITION_1024": "0",
    },
    "mfma4_all_512": {
        "VLLM_ROCM_FUSED_SHORT_SEQ": "1",
        "VLLM_ROCM_MFMA4_ALL": "1",
        "VLLM_ROCM_PARTITION_512": "1",
        "VLLM_ROCM_PARTITION_1024": "0",
    },
    "mfma4_all_1024": {
        "VLLM_ROCM_FUSED_SHORT_SEQ": "1",
        "VLLM_ROCM_MFMA4_ALL": "1",
        "VLLM_ROCM_PARTITION_512": "0",
        "VLLM_ROCM_PARTITION_1024": "1",
    },
}

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Prompts of different lengths to test different attention scenarios
PROMPTS = {
    "short": {
        "text": "Hello, my name is",
        "max_tokens": 128,
    },
    "medium": {
        "text": (
            "Write a detailed explanation of how attention mechanisms work "
            "in transformer neural networks. Cover the key concepts of "
            "queries, keys, and values, and explain how self-attention "
            "allows the model to weigh the importance of different parts "
            "of the input sequence."
        ),
        "max_tokens": 256,
    },
    "long": {
        "text": (
            "Summarize the following text in detail: " + "word " * 250
        ),
        "max_tokens": 512,
    },
}

# ---------------------------------------------------------------------------
# Worker script for model inference
# ---------------------------------------------------------------------------
MODEL_WORKER_SCRIPT = r'''
import json
import sys

args = json.loads(sys.argv[1])

from vllm import LLM, SamplingParams

llm = LLM(
    model=args["model"],
    dtype="bfloat16",
    max_model_len=2048,
    enforce_eager=True,
    seed=42,
)

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=args["max_tokens"],
    seed=42,
)

outputs = llm.generate(args["prompts"], sampling_params)

result = {
    "token_ids": [list(out.outputs[0].token_ids) for out in outputs],
    "texts": [out.outputs[0].text for out in outputs],
}

with open(args["output_path"], "w") as f:
    json.dump(result, f)
'''


def run_model_subprocess(mode_name, env_vars, prompts, max_tokens,
                         output_path):
    """Run model inference in a subprocess with the given env vars."""
    env = os.environ.copy()
    env.update(env_vars)

    args_json = json.dumps({
        "model": MODEL,
        "prompts": prompts,
        "max_tokens": max_tokens,
        "output_path": output_path,
    })

    result = subprocess.run(
        [sys.executable, "-c", MODEL_WORKER_SCRIPT, args_json],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Model inference failed for mode '{mode_name}':\n"
            f"stderr: {result.stderr[-1500:]}"
        )

    with open(output_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("prompt_category", ["short", "medium", "long"])
def test_model_output_matches_across_modes(prompt_category, tmp_path):
    """All kernel modes must produce identical greedy-decoded tokens."""
    prompt_config = PROMPTS[prompt_category]
    prompts = [prompt_config["text"]]
    max_tokens = prompt_config["max_tokens"]

    results = {}
    for mode_name, env_vars in KERNEL_MODES.items():
        output_path = str(tmp_path / f"model_{mode_name}.json")
        results[mode_name] = run_model_subprocess(
            mode_name, env_vars, prompts, max_tokens, output_path,
        )

    # Compare all modes against baseline
    baseline_tokens = results["baseline"]["token_ids"]
    baseline_text = results["baseline"]["texts"]

    for mode_name, data in results.items():
        if mode_name == "baseline":
            continue

        assert data["token_ids"] == baseline_tokens, (
            f"\nMode '{mode_name}' produced different tokens than baseline "
            f"for '{prompt_category}' prompt.\n"
            f"Baseline tokens (first 20): {baseline_tokens[0][:20]}\n"
            f"{mode_name} tokens (first 20): {data['token_ids'][0][:20]}\n"
            f"Baseline text (first 100): {baseline_text[0][:100]}\n"
            f"{mode_name} text (first 100): {data['texts'][0][:100]}"
        )


def test_all_modes_produce_valid_output(tmp_path):
    """Each kernel mode must produce non-empty, reasonable output."""
    prompts = ["What is 2 + 2?"]
    max_tokens = 32

    for mode_name, env_vars in KERNEL_MODES.items():
        output_path = str(tmp_path / f"model_{mode_name}.json")
        result = run_model_subprocess(
            mode_name, env_vars, prompts, max_tokens, output_path,
        )

        tokens = result["token_ids"][0]
        text = result["texts"][0]

        assert len(tokens) > 0, (
            f"Mode '{mode_name}' produced empty output"
        )
        assert len(text.strip()) > 0, (
            f"Mode '{mode_name}' produced empty text"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
