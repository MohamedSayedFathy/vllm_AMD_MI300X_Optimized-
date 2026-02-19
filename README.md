<!-- markdownlint-disable MD001 MD041 -->
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
Easy, fast, and cheap LLM serving for everyone
</h3>

<p align="center">
| <a href="https://docs.vllm.ai"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> |
</p>

🔥 We have built a vllm website to help you get started with vllm. Please visit [vllm.ai](https://vllm.ai) to learn more.
For events, please visit [vllm.ai/events](https://vllm.ai/events) to join us.

---

## AMD MI300X: Optimized PagedAttention v2 Kernel

This fork extends vLLM with a custom PagedAttention v2 kernel optimized for AMD MI300X GPUs, delivering up to **17.6% higher throughput** on real LLM workloads.

### Optimization Overview

Five kernel modes are implemented and benchmarkable:

| Mode | Description |
|------|-------------|
| `baseline` | Unmodified vLLM ROCm kernel |
| `fused` | Single-partition sequences skip the reduce pass |
| `mfma4_all` | MFMA4 instruction path for all GQA ratios (256-token partitions) |
| `mfma4_all_512` | MFMA4 with 512-token partitions |
| `mfma4_all_1024` | MFMA4 with 1024-token partitions (best for GQA ≤ 12) |

Modes are selected via environment variables — no recompilation needed:

```bash
# baseline (default)
python script.py

# fused short-seq
VLLM_ROCM_FUSED_SHORT_SEQ=1 python script.py

# mfma4_all (256-partition)
VLLM_ROCM_MFMA4_ALL=1 python script.py

# mfma4_all_512
VLLM_ROCM_MFMA4_ALL=1 VLLM_ROCM_PARTITION_512=1 python script.py

# mfma4_all_1024
VLLM_ROCM_MFMA4_ALL=1 VLLM_ROCM_PARTITION_1024=1 python script.py
```

### Requirements

- AMD MI300X GPU with ROCm 6.x
- vLLM built from source (this repo)
- Python 3.10+
- HuggingFace access token for gated models (Llama, Mistral)

### Build from Source (ROCm)

```bash
git clone https://github.com/MohamedSayedFathy/TUM_Practical-_Course.git
cd TUM_Practical-_Course

pip install -e ".[rocm]"
```

### Running Correctness Tests

Two test suites verify that all kernel modes produce identical outputs to the baseline.

**Unit test** — runs the attention kernel directly on synthetic tensors:

```bash
pytest tests/test_kernel_correctness.py -v
```

**Application test** — runs TinyLlama end-to-end with greedy decoding and compares token outputs:

```bash
pytest tests/test_model_correctness.py -v
```

### Running Benchmarks

All benchmarks are in `benchmark_fused_short_seq.py`. Each mode is run in a separate subprocess to avoid env-var caching issues.

#### Kernel Microbenchmark

Measures raw attention kernel latency across GQA ratios and sequence lengths:

```bash
python benchmark_fused_short_seq.py kernel --preset all
```

#### Throughput Benchmark (Real Workload)

Measures end-to-end tokens/second using ShareGPT conversation data:

```bash
# Single model
python benchmark_fused_short_seq.py throughput \
  --models meta-llama/Llama-3.1-8B-Instruct \
  --num-prompts 500 \
  --max-model-len 12000 \
  --enforce-eager

# All three models
python benchmark_fused_short_seq.py throughput \
  --models meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen2.5-7B-Instruct mistralai/Mistral-7B-Instruct-v0.3 \
  --num-prompts 500 \
  --max-model-len 12000 \
  --enforce-eager
```

#### Latency Sweep

Measures per-batch latency across input lengths, output lengths, and batch sizes:

```bash
python benchmark_fused_short_seq.py latency_sweep \
  --models meta-llama/Llama-3.1-8B-Instruct \
  --input-lens 500 1000 4000 8000 12000 \
  --output-lens 128 \
  --batch-sizes 32 64 128 256 \
  --max-model-len 16384 \
  --enforce-eager
```

#### Display Saved Results

```bash
# Show throughput results
python benchmark_fused_short_seq.py display --file benchmark_results/throughput_*.json

# Show latency results
python benchmark_fused_short_seq.py display --file benchmark_results/latency_sweep_*.json
```

---

## About

vLLM is a fast and easy-to-use library for LLM inference and serving.

Originally developed in the [Sky Computing Lab](https://sky.cs.berkeley.edu) at UC Berkeley, vLLM has evolved into a community-driven project with contributions from both academia and industry.

vLLM is fast with:

- State-of-the-art serving throughput
- Efficient management of attention key and value memory with [**PagedAttention**](https://blog.vllm.ai/2023/06/20/vllm.html)
- Continuous batching of incoming requests
- Fast model execution with CUDA/HIP graph
- Quantizations: [GPTQ](https://arxiv.org/abs/2210.17323), [AWQ](https://arxiv.org/abs/2306.00978), [AutoRound](https://arxiv.org/abs/2309.05516), INT4, INT8, and FP8
- Optimized CUDA kernels, including integration with FlashAttention and FlashInfer
- Speculative decoding
- Chunked prefill

vLLM is flexible and easy to use with:

- Seamless integration with popular Hugging Face models
- High-throughput serving with various decoding algorithms, including *parallel sampling*, *beam search*, and more
- Tensor, pipeline, data and expert parallelism support for distributed inference
- Streaming outputs
- OpenAI-compatible API server
- Support for NVIDIA GPUs, AMD CPUs and GPUs, Intel CPUs and GPUs, PowerPC CPUs, Arm CPUs, and TPU. Additionally, support for diverse hardware plugins such as Intel Gaudi, IBM Spyre and Huawei Ascend.
- Prefix caching support
- Multi-LoRA support

vLLM seamlessly supports most popular open-source models on HuggingFace, including:

- Transformer-like LLMs (e.g., Llama)
- Mixture-of-Expert LLMs (e.g., Mixtral, Deepseek-V2 and V3)
- Embedding Models (e.g., E5-Mistral)
- Multi-modal LLMs (e.g., LLaVA)

Find the full list of supported models [here](https://docs.vllm.ai/en/latest/models/supported_models.html).

## Getting Started

Install vLLM with `pip` or [from source](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#build-wheel-from-source):

```bash
pip install vllm
```

Visit our [documentation](https://docs.vllm.ai/en/latest/) to learn more.

- [Installation](https://docs.vllm.ai/en/latest/getting_started/installation.html)
- [Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
- [List of Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

## Contributing

We welcome and value any contributions and collaborations.
Please check out [Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/index.html) for how to get involved.

## Citation

If you use vLLM for your research, please cite our [paper](https://arxiv.org/abs/2309.06180):

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```

## Contact Us

<!-- --8<-- [start:contact-us] -->
- For technical questions and feature requests, please use GitHub [Issues](https://github.com/vllm-project/vllm/issues)
- For discussing with fellow users, please use the [vLLM Forum](https://discuss.vllm.ai)
- For coordinating contributions and development, please use [Slack](https://slack.vllm.ai)
- For security disclosures, please use GitHub's [Security Advisories](https://github.com/vllm-project/vllm/security/advisories) feature
- For collaborations and partnerships, please contact us at [collaboration@vllm.ai](mailto:collaboration@vllm.ai)
<!-- --8<-- [end:contact-us] -->

## Media Kit

- If you wish to use vLLM's logo, please refer to [our media kit repo](https://github.com/vllm-project/media-kit)
