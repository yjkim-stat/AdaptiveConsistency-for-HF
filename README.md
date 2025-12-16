## Overview

**Adaptive-Consistency (AC)** is a cost-efficient alternative to Self-Consistency decoding.
Instead of sampling a fixed number of reasoning paths per query, AC **dynamically allocates the sampling budget** based on the agreement observed so far, stopping early when sufficient confidence is reached.

The **original implementation** focuses primarily on API-based LLMs (e.g., OpenAI).
This repository **extends Adaptive-Consistency to Hugging Face**, making it practical for:

* Local inference with open-weight models (LLaMA, Gemma, Qwen, Mistral, etc.)
* Offline or privacy-sensitive settings
* Large-scale evaluation pipelines using `transformers` and `datasets`
* Research on decoding, reasoning, and test-time scaling

---

## Key Features of This Repository

### 🚀 Hugging Face Native Support

* Compatible with `AutoModelForCausalLM` and `AutoTokenizer`
* Supports local GPU inference and Hugging Face caching (`cache_dir`)
* Easily integrates with existing `transformers.generate()` pipelines

### 🔁 Model-Agnostic Adaptive Sampling

* Works with **any causal LM** that can generate multiple samples
* No training or fine-tuning required
* Drop-in replacement for standard Self-Consistency loops

---

## Differences from the Original Repository

| Aspect                      | Original Repo     | This Repo                           |
| --------------------------- | ----------------- | ----------------------------------- |
| LLM Backend                 | OpenAI-style APIs | **Hugging Face Transformers**       |
| Local Models                | ❌                 | **✅ Supported**                     |
| Offline Inference           | ❌                 | **✅ Supported**                     |
| Research Decoding Pipelines | Limited           | **Designed for research workflows** |

> This implementation preserves the **algorithmic logic, stopping criteria, and evaluation protocol** of the original Adaptive-Consistency paper while extending its applicability to modern open-source LLM ecosystems.

---

## Citation

If you use this code, please cite the original paper:

```bibtex
@misc{aggarwal2023lets,
  title={Let's Sample Step by Step: Adaptive-Consistency for Efficient Reasoning with LLMs},
  author={Pranjal Aggarwal and Aman Madaan and Yiming Yang and Mausam},
  year={2023},
  eprint={2305.11860},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
```

---

## Acknowledgements

This repository is built upon the original **Adaptive-Consistency** implementation by
Pranjal Aggarwal et al.
All credit for the core algorithm and experimental insights belongs to the original authors.