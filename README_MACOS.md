# OmniGen2

OmniGen2 is a multimodal generative AI model that marries state-of-the-art visual understanding with high-quality text-to-image generation, image editing, and in-context multimodal reasoning.

This repository contains two tightly-integrated code paths:

1. **Cross-platform reference implementation** — works on Linux/Windows with NVIDIA GPUs (CUDA).
2. **🍎 Apple Silicon fork** — optimised for M-series Macs using the Metal Performance Shaders (MPS) backend.

**TL;DR** – If you are on an M1/M2/M3/M4 Mac, jump to the Apple Silicon Quick-Start. Everyone else can follow the standard CUDA Quick-Start.

---

## Table of Contents

1. [Key Features](#key-features)
2. [Architecture Overview](#architecture-overview)
3. [Environment Setup](#environment-setup)
   - [CUDA (NVIDIA GPU)](#cuda-nvidia-gpu)
   - [Apple Silicon (MPS)](#apple-silicon-mps)
4. [Running Examples](#running-examples)
5. [Gradio Applications](#gradio-applications)
6. [Direct Inference](#direct-inference)
7. [Key Generation Parameters](#key-generation-parameters)
8. [Performance Optimisation & Benchmarks](#performance-optimisation--benchmarks)
9. [Troubleshooting](#troubleshooting)
10. [Contributing](#contributing)
11. [License](#license)

---

## Key Features

- **Dual decoding pathways** for text & image generation built on a Qwen-VL-2.5 foundation.
- **Four core capabilities:**
  1. Visual understanding & analysis
  2. High-fidelity text-to-image generation
  3. Instruction-guided image editing
  4. In-context multimodal generation (RAG-style image & text prompts)
- **Modular pipeline architecture** — swap schedulers, attention processors, and embeddings with minimal code changes.
- **🍎 Apple Silicon enhancements** (M-series Macs):
  - Native MPS backend (no Docker, no drivers)
  - Flash-Attention removed / Triton ops eliminated
  - Float32 precision enforced for maximum compatibility
  - Drop-in Gradio UI with lighter theme, hover effects, and mobile responsiveness

---

## Architecture Overview

```
omnigen2/
├─ models/                 # Core model architectures
│   ├─ transformer_omnigen2.py
│   ├─ embeddings.py
│   └─ attention_processor.py
├─ pipelines/omnigen2/     # Main inference pipelines
│   ├─ pipeline_omnigen2.py           # Image generation
│   └─ pipeline_omnigen2_chat.py      # Multimodal chat
├─ schedulers/             # Diffusion & flow-matching schedulers
│   ├─ scheduling_flow_match_euler_discrete.py
│   └─ scheduling_dpmsolver_multistep.py
└─ utils/                  # Misc utilities
    └─ ...
```

A two-stage loading pattern is used throughout:

1. Load the base pipeline with `trust_remote_code=False` (local ops only).
2. Load each large component (Transformer, VAE, Scheduler, Vision Encoder) manually.
3. Optionally apply CPU or sequential CPU offload to cut VRAM requirements by 50–80%.

---

## Environment Setup

### 📦 Common Steps

```bash
# Clone the repository
git clone https://github.com/zettai-seigi/OmniGen2.git
cd OmniGen2

# Create an isolated environment
conda create -n omnigen2 python=3.11
conda activate omnigen2

# Install Python requirements shared by ALL platforms
pip install -r requirements.txt
```

### 🚀 Quick Start for CUDA

```bash
# Install PyTorch w/ CUDA 12.4 (adjust for your toolkit)
pip install torch==2.6.0 torchvision --extra-index-url https://download.pytorch.org/whl/cu124
```

**Tip:** Do not install this CUDA build on Apple Silicon — see below.

### 🚀 Quick Start for Apple Silicon

```bash
# Install the official MPS build of PyTorch
pip install torch==2.6.0 torchvision torchaudio

# ⚠️ flash-attn & triton are **NOT** required and will fail to build on macOS.
```

---

## Running Examples

```bash
# 1. Text-to-image
python inference.py \ 
  --model_path OmniGen2/OmniGen2 \
  --instruction "A beautiful landscape with mountains at sunset" \
  --output_image_path outputs/landscape.png

# 2. Image editing (reference + instruction)
python inference.py \
  --model_path OmniGen2/OmniGen2 \
  --instruction "Add a bird to this scene" \
  --input_image_path path/to/scene.jpg \
  --output_image_path outputs/scene_with_bird.png

# 3. Visual understanding / multimodal chat
python inference_chat.py \
  --model_path OmniGen2/OmniGen2 \
  --instruction "Describe the objects in this image" \
  --input_image_path path/to/image.jpg
```

---

## Gradio Applications

| Command | What it does |
|---------|--------------|
| `python app_chat_enhanced.py` | ✨ Enhanced Apple-first UI with modern styling |

---

## Direct Inference

The CLI wrappers expose all runtime knobs. Below are the most commonly tuned flags:

```bash
# Basic inference (Mac, float32 by default)
python inference.py \
  --model_path OmniGen2/OmniGen2 \
  --dtype fp32 \
  --instruction "Your prompt here" \
  --output_image_path outputs/result.png

# CPU offload for limited VRAM (Mac / laptop GPU)
python inference.py \
  --enable_model_cpu_offload \
  --instruction "Your prompt here" \
  --output_image_path outputs/result.png
```

---

## Key Generation Parameters

| Flag | Default | Effect |
|------|---------|--------|
| `text_guidance_scale` | 4.0 – 5.0 | Higher = closer adherence to prompt text |
| `image_guidance_scale` | 1.2 – 3.0 | Higher = closer adherence to reference image |
| `num_inference_steps` | 50 | Denoising iterations (30–60 is common) |
| `max_pixels` | 1024×1024 | Auto-resizing cap to avoid OOM |
| `enable_model_cpu_offload` | False | Swap layers to CPU to halve VRAM usage |
| `enable_sequential_cpu_offload` | False | Extreme offload (<3 GB VRAM, slower) |

### Apple Silicon preset

```python
dtype = "fp32"             # MPS requires float32
enable_model_cpu_offload = True
text_guidance_scale = 5.0
image_guidance_scale = 2.0
num_inference_steps = 50
```

---

## Performance Optimisation & Benchmarks

### Apple Silicon (M3 Max • 36 GB RAM)

| Task | Resolution | Steps | Time | Peak Mem |
|------|------------|-------|------|----------|
| Text-to-Image | 1024×1024 | 50 | 45-60 s | 12-15 GB |
| Image Editing | 1024×1024 | 50 | 40-55 s | 12-15 GB |
| ↳ with CPU offload | 1024×1024 | 50 | +15-20% | 8-10 GB |

### Performance Tips

- **16 GB+ unified memory?** Disable offload for ~10-15% speed boost.
- **Under 8 GB?** Use `--enable_sequential_cpu_offload`.
- **Reduce** `num_inference_steps` (e.g. 30) for faster previews.

---

## Troubleshooting

<details>
<summary>Common errors & fixes</summary>

| Message | Likely Cause | Fix |
|---------|--------------|-----|
| `ModuleNotFoundError: triton` | Triton kernels removed on Mac | You are on the Apple fork → no action needed |
| `RuntimeError: MPS tensor has incompatible dtype` | Mixed dtypes | Always run with `--dtype fp32` |
| Flash-attention import errors | CUDA-only dep present | Ensure flash-attn is not installed on Mac |
| VRAM OOM | High-res images, many steps | Use offload flags or lower max_pixels |

</details>

---

## Contributing

- **CUDA & cross-platform issues** → original VectorSpaceLab/OmniGen2.
- **Apple Silicon specific issues** → this fork: zettai-seigi/OmniGen2.

We welcome PRs that improve documentation, add schedulers, or optimise MPS kernels.

---

## License

This project inherits the licence of the upstream OmniGen2 repository. See LICENSE for details.

---

© 2025 – OmniGen2 Contributors • Optimised for Apple Silicon by zettai-seigi