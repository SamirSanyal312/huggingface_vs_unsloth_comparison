# Hugging Face vs Unsloth Fine-Tuning Comparison

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)]()
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository compares two approaches for supervised fine-tuning a small language model:

1. **Hugging Face + PEFT + bitsandbytes + TRL**
2. **Unsloth + TRL**

The goal is to evaluate training speed, GPU memory usage, and inference throughput under a controlled setup using the same base model, dataset, LoRA configuration, and training hyperparameters.

---

## Repository Contents

```text
.
├── hugging_face_solution.ipynb   # Hugging Face / PEFT / bitsandbytes fine-tuning workflow
├── Unsloth_solution.ipynb        # Unsloth fine-tuning workflow
├── README.md                     # Project documentation
├── requirements.txt              # Python dependencies
├── LICENSE                       # MIT License
└── .gitignore                    # Files and folders excluded from Git
```

---

## Project Objective

Fine-tuning large language models can be expensive in terms of time and GPU memory. This project compares a standard Hugging Face QLoRA-style workflow with an Unsloth-based workflow to understand the practical differences in:

- Training time
- Peak allocated GPU memory
- Peak reserved GPU memory
- Inference tokens per second
- Code complexity and reproducibility

---

## Experiment Setup

Both notebooks use the same experimental configuration wherever possible.

| Component | Configuration |
|---|---|
| Base model | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| Dataset | `yahma/alpaca-cleaned` |
| Dataset size | First 2,000 examples |
| Task | Supervised fine-tuning on instruction-response data |
| Max sequence length | 1,024 |
| Training steps | 30 |
| Batch size | 2 |
| Gradient accumulation | 4 |
| Learning rate | `2e-4` |
| LoRA rank | 16 |
| LoRA alpha | 16 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Optimizer | `paged_adamw_8bit` |
| GPU used in notebook run | Tesla T4 |

---

## Results From Current Notebook Run

The following values are from the executed notebooks in this repository. Results may vary depending on GPU type, CUDA version, package versions, and runtime conditions.

| Metric | Hugging Face Workflow | Unsloth Workflow |
|---|---:|---:|
| Training time | 80.00 sec | 65.47 sec |
| Peak allocated VRAM | 1.756 GB | 0.952 GB |
| Peak reserved VRAM | 2.199 GB | 1.693 GB |
| Inference throughput | 13.03 tokens/sec | 25.29 tokens/sec |

### Observed Difference

| Comparison | Result |
|---|---:|
| Training speedup with Unsloth | ~1.22x faster |
| Allocated VRAM reduction with Unsloth | ~45.8% lower |
| Reserved VRAM reduction with Unsloth | ~23.0% lower |
| Inference throughput improvement with Unsloth | ~1.94x faster |

In this specific run, Unsloth trained faster, used less allocated GPU memory, and produced higher inference throughput than the standard Hugging Face workflow.

---

## Notebooks

### 1. `hugging_face_solution.ipynb`

This notebook implements fine-tuning using the Hugging Face ecosystem:

- `transformers`
- `datasets`
- `peft`
- `bitsandbytes`
- `trl`

Main steps:

1. Load the Alpaca-cleaned dataset
2. Format instruction-response examples
3. Load TinyLlama in 4-bit quantized mode
4. Prepare the model for k-bit training
5. Add LoRA adapters using PEFT
6. Fine-tune using `SFTTrainer`
7. Measure training time, VRAM usage, and inference speed

### 2. `Unsloth_solution.ipynb`

This notebook implements the same fine-tuning task using Unsloth:

- `unsloth`
- `datasets`
- `trl`
- `transformers`

Main steps:

1. Load the Alpaca-cleaned dataset
2. Format instruction-response examples
3. Load TinyLlama through `FastLanguageModel`
4. Add LoRA adapters using Unsloth
5. Fine-tune using `SFTTrainer`
6. Enable Unsloth inference optimizations
7. Measure training time, VRAM usage, and inference speed

---

## How to Run

### Option 1: Run in Google Colab

This is the recommended option because the notebooks require a CUDA-enabled GPU.

1. Open either notebook in Google Colab.
2. Go to **Runtime > Change runtime type**.
3. Select **GPU**.
4. Run the cells from top to bottom.
5. Compare the printed `FINAL RESULTS` section from both notebooks.

### Option 2: Run Locally

A local NVIDIA GPU with CUDA support is recommended.

```bash
git clone https://github.com/SamirSanyal312/huggingface_vs_unsloth_comparison.git
cd huggingface_vs_unsloth_comparison

python -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\activate     # Windows PowerShell

pip install -r requirements.txt
jupyter notebook
```

Then open and run:

```text
hugging_face_solution.ipynb
Unsloth_solution.ipynb
```

> Note: PyTorch installation can vary by CUDA version. If `torch` installation fails locally, install the correct PyTorch build for your system first, then install the remaining requirements.

---

## Reproducibility Notes

For a fair comparison, both notebooks keep the following settings aligned:

- Same base model
- Same dataset
- Same number of training examples
- Same prompt format
- Same LoRA rank and alpha
- Same target attention modules
- Same number of training steps
- Same optimizer
- Same inference prompt

Small differences can still occur because Unsloth applies internal optimizations and patches that are different from the standard Hugging Face training path.

---

## Key Takeaways

- Hugging Face provides a flexible and widely used fine-tuning workflow.
- Unsloth can significantly reduce memory usage and improve speed for supported models and hardware.
- For small GPU environments such as Colab T4, Unsloth can be a practical option for faster experimentation.
- For production decisions, this experiment should be repeated across larger datasets, more steps, different GPUs, and held-out evaluation prompts.

---

## Future Improvements

Potential next steps:

- Add a benchmark script that runs both workflows automatically
- Save results to a CSV file for easier comparison
- Add held-out evaluation prompts
- Compare model quality after fine-tuning
- Test larger models such as Llama 3, Mistral, or Qwen variants
- Compare adapter-only inference versus merged-model inference
- Add Weights & Biases or TensorBoard logging
- Run the experiment on A100/L4 GPUs and compare with T4 results

---

## Author

**Samir Sanyal**  
GitHub: [@SamirSanyal312](https://github.com/SamirSanyal312)

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
