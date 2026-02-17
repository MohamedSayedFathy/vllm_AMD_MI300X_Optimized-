#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for PagedAttention v2 kernel mode correctness (AMD MI300X).

Tests that all 5 kernel modes produce identical attention outputs:
  1. baseline       - Original 2-kernel path (no fused)
  2. fused          - Skip reduce for short seqs (seq <= 256)
  3. mfma4_all      - MFMA4 for all GQA ratios + fused
  4. mfma4_all_512  - MFMA4 all GQA + 512-token partition
  5. mfma4_all_1024 - MFMA4 all GQA + 1024-token partition

Each mode runs in a separate subprocess because the env vars that control
kernel behavior are cached in static C++ variables (csrc/rocm/attention.cu).
"""

import json
import os
import random
import subprocess
import sys
import tempfile

import pytest
import torch

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

# ---------------------------------------------------------------------------
# Worker script executed in each subprocess
# ---------------------------------------------------------------------------
WORKER_SCRIPT = r'''
import json
import os
import random
import sys
import torch

# Parse arguments
args = json.loads(sys.argv[1])

num_seqs = args["num_seqs"]
num_query_heads = args["num_query_heads"]
num_kv_heads = args["num_kv_heads"]
head_size = args["head_size"]
block_size = args["block_size"]
seq_len = args["seq_len"]
seed = args["seed"]
output_path = args["output_path"]

NUM_BLOCKS = 4321
PARTITION_SIZE = 256
dtype = torch.bfloat16
device = "cuda"

# Set deterministic random state
torch.manual_seed(seed)
random.seed(seed)
torch.set_default_device(device)

scale = float(1.0 / (head_size ** 0.5))

# Create query tensor
query = torch.empty(num_seqs, num_query_heads, head_size, dtype=dtype)
query.uniform_(-scale, scale)

# Create sequence lengths and block tables
seq_lens_list = [seq_len] * num_seqs
max_seq_len = seq_len
seq_lens = torch.tensor(seq_lens_list, dtype=torch.int)

max_num_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
block_tables_lst = []
for _ in range(num_seqs):
    block_table = [random.randint(0, NUM_BLOCKS - 1)
                   for _ in range(max_num_blocks_per_seq)]
    block_tables_lst.append(block_table)
block_tables = torch.tensor(block_tables_lst, dtype=torch.int)

# Create KV caches (same pattern as test_attention.py)
from vllm.utils.torch_utils import create_kv_caches_with_random
key_caches, value_caches = create_kv_caches_with_random(
    NUM_BLOCKS, block_size, 1, num_kv_heads, head_size,
    "auto", dtype, seed, device)
key_cache, value_cache = key_caches[0], value_caches[0]

k_scale = v_scale = torch.tensor(1.0, dtype=torch.float32, device=device)

# Allocate output tensors
num_partitions = (max_seq_len + PARTITION_SIZE - 1) // PARTITION_SIZE
output = torch.empty_like(query)
tmp_output = torch.empty(
    num_seqs, num_query_heads, num_partitions, head_size, dtype=dtype)
exp_sums = torch.empty(
    num_seqs, num_query_heads, num_partitions, dtype=torch.float32)
max_logits = torch.empty_like(exp_sums)

# Call the ROCm paged attention kernel
from vllm import _custom_ops as ops
ops.paged_attention_rocm(
    output, exp_sums, max_logits, tmp_output,
    query, key_cache, value_cache,
    num_kv_heads, scale, block_tables, seq_lens,
    None,  # query_start_loc
    block_size, max_seq_len,
    None,  # alibi_slopes
    "auto",  # kv_cache_dtype
    k_scale, v_scale,
)

# Save output to file
torch.save(output.cpu(), output_path)
'''


def run_kernel_subprocess(mode_name, env_vars, num_seqs, num_query_heads,
                          num_kv_heads, head_size, block_size, seq_len,
                          seed, output_path):
    """Run the kernel in a subprocess with the given env vars."""
    env = os.environ.copy()
    env.update(env_vars)

    args_json = json.dumps({
        "num_seqs": num_seqs,
        "num_query_heads": num_query_heads,
        "num_kv_heads": num_kv_heads,
        "head_size": head_size,
        "block_size": block_size,
        "seq_len": seq_len,
        "seed": seed,
        "output_path": output_path,
    })

    result = subprocess.run(
        [sys.executable, "-c", WORKER_SCRIPT, args_json],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Subprocess failed for mode '{mode_name}':\n"
            f"stderr: {result.stderr[-1000:]}"
        )

    return torch.load(output_path, weights_only=True)


# ---------------------------------------------------------------------------
# Test parameters
# ---------------------------------------------------------------------------
GQA_CONFIGS = [
    (32, 8),   # GQA=4  (Llama-3.1-8B style)
    (28, 4),   # GQA=7  (Qwen2.5-7B style)
    (32, 4),   # GQA=8
]

# These seq lengths cross the fused boundaries for different partition sizes:
# 128, 256 = single partition (fused for all modes)
# 512      = fused boundary for mfma4_all_512
# 1024     = fused boundary for mfma4_all_1024
# 2048     = multi-partition for all modes
SEQ_LENGTHS = [128, 256, 512, 1024, 2048]

BATCH_SIZES = [1, 4, 16]
HEAD_SIZE = 128
BLOCK_SIZE = 16
SEED = 42


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("num_heads", GQA_CONFIGS,
                         ids=[f"GQA{q // k}_{q}q{k}kv" for q, k in GQA_CONFIGS])
@pytest.mark.parametrize("seq_len", SEQ_LENGTHS,
                         ids=[f"seq{s}" for s in SEQ_LENGTHS])
@pytest.mark.parametrize("num_seqs", BATCH_SIZES,
                         ids=[f"batch{b}" for b in BATCH_SIZES])
def test_all_modes_match_baseline(num_seqs, seq_len, num_heads, tmp_path):
    """All optimized kernel modes must produce outputs matching baseline."""
    num_query_heads, num_kv_heads = num_heads

    # Run all modes and collect outputs
    outputs = {}
    for mode_name, env_vars in KERNEL_MODES.items():
        output_path = str(tmp_path / f"output_{mode_name}.pt")
        outputs[mode_name] = run_kernel_subprocess(
            mode_name, env_vars,
            num_seqs, num_query_heads, num_kv_heads,
            HEAD_SIZE, BLOCK_SIZE, seq_len, SEED,
            output_path,
        )

    # Compare all modes against baseline
    baseline = outputs["baseline"]
    for mode_name, output in outputs.items():
        if mode_name == "baseline":
            continue
        torch.testing.assert_close(
            output, baseline,
            atol=1e-3, rtol=1e-3,
            msg=f"Mode '{mode_name}' diverges from baseline "
                f"(seq_len={seq_len}, batch={num_seqs}, "
                f"GQA={num_query_heads // num_kv_heads})",
        )


@pytest.mark.parametrize("seq_len", [256, 1024],
                         ids=[f"seq{s}" for s in [256, 1024]])
def test_fused_vs_unfused_boundary(seq_len, tmp_path):
    """Test specifically at the fused/unfused boundary for each mode."""
    num_query_heads, num_kv_heads = 32, 8  # GQA=4
    num_seqs = 4

    outputs = {}
    for mode_name, env_vars in KERNEL_MODES.items():
        output_path = str(tmp_path / f"output_{mode_name}.pt")
        outputs[mode_name] = run_kernel_subprocess(
            mode_name, env_vars,
            num_seqs, num_query_heads, num_kv_heads,
            HEAD_SIZE, BLOCK_SIZE, seq_len, SEED,
            output_path,
        )

    baseline = outputs["baseline"]
    for mode_name, output in outputs.items():
        if mode_name == "baseline":
            continue
        torch.testing.assert_close(
            output, baseline,
            atol=1e-3, rtol=1e-3,
            msg=f"Mode '{mode_name}' diverges at boundary seq_len={seq_len}",
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
