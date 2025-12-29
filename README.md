# Jasper & Stella: Distillation of SOTA Embedding Models

A faithful PyTorch implementation of the research paper "Jasper and Stella: Distillation of SOTA embedding models" (arXiv:2412.19048v2).

This repository implements the Jasper student model (2B parameters) and the novel 4-Stage Distillation Framework used to distill knowledge from multiple larger teacher models (Stella 1.5B and NV-Embed-v2) into a compact, high-performance, and multimodal embedding model.

---

## Key Features

*   **Dual-Teacher Distillation:** Implements the concatenation and normalization strategy to combine `dunzhang/stella_en_1.5B_v5` and `nvidia/NV-Embed-v2` into a single "Ground Truth" target.
*   **Matryoshka Representation Learning (MRL):** Supports flexible embedding dimensions (12288, 1024, 512, 256) via multiple projection heads.
*   **Triple-Loss Objective:** Implements the paper's specific loss formulation:
    1.  **Cosine Loss** ($\mathcal{L}_{cosine}$): Aligns absolute vector direction.
    2.  **Similarity Loss** ($\mathcal{L}_{sim}$): Aligns the student's semantic similarity matrix with the teacher's.
    3.  **Relative Similarity Loss** ($\mathcal{L}_{resim}$): A ranking-based loss ensuring the student preserves the teacher's relative preferences.
*   **Multimodal Architecture:** Integrates a SigLIP Vision Encoder to align image features with the text embedding space (Stage 4).
*   **4-Stage Training Pipeline:** Automates the complex parameter freezing and unfreezing schedule described in the paper.

---

## Project Structure

```bash
jasper_impl/
├── src/
│   ├── model.py      # The JasperModel architecture (LLM + Vision + MRL Heads)
│   ├── teachers.py   # DualTeacherWrapper (Manages Stella + NV-Embed)
│   ├── losses.py     # Custom DistillationLoss implementation
│   ├── train.py      # Main training loop implementing the 4 stages
│   └── utils.py      # Data simulation utilities
└── requirements.txt  # Dependencies
```

---

## Quick Start (Simulation Mode)

By default, this repository runs in Simulation Mode. This allows you to verify the entire pipeline (architecture instantiation, forward passes, loss calculation, backward propagation, and stage transitions) without downloading 20GB+ of model weights.

In this mode:
*   Models are initialized with Random Weights (using "Tiny" config variations for speed).
*   Data is generated as random tensor noise.

**1. Install Dependencies:**
```bash
pip install -r requirements.txt
```

**2. Run the Verification Script:**
```bash
python src/train.py
```
*You will see the loss decrease over 5 steps for each of the 4 stages, confirming the logic works end-to-end.*

---

## Switching to Production (Real Training)

To use this code for actual model training, you need to enable "Real Mode".

**Prerequisites:**
*   **GPU:** At least 24GB VRAM (ideally 48GB+ or multi-GPU) to hold the Teachers (frozen) and Student (trainable).
*   **Storage:** ~20GB free space for model checkpoints.

**Steps:**

1.  **Enable Real Weights:**
    Open `src/train.py` and update the initialization:
    ```python
    # Change load_real_weights to True
    teacher = DualTeacherWrapper(load_real_weights=True)
    ```
    *This will trigger the download of `dunzhang/stella_en_1.5B_v5` and `nvidia/NV-Embed-v2`.*

2.  **Load Real Data:**
    Open `src/utils.py` or modify the loop in `src/train.py`. Replace `get_dummy_batch()` with a real PyTorch `DataLoader` serving your dataset (e.g., FineWeb-Edu or MTEB).
    ```python
    # Example pseudo-code in train.py
    # dataset = load_dataset("HuggingFaceFW/fineweb-edu", split="train")
    # dataloader = DataLoader(dataset, batch_size=128, ...)
    # for batch in dataloader: ...
    ```

3.  **Enable Full Architecture:**
    If you modified `src/teachers.py` to use "Tiny" configs for verification, revert those changes to the standard `AutoConfig` calls to ensure full-sized models are loaded.

---

## Methodology Details

This implementation strictly follows the paper's 4-stage process:

| Stage | Goal | Trainable Parameters | Description |
| :--- | :--- | :--- | :--- |
| **1** | **Distillation** | **FC1 Only** | The student learns to map its output to the concatenated teacher space (12,288 dim). |
| **2** | **Fine-Tuning** | **FC1 + Last 3 Layers** | The student's encoder is refined to better support the high-dimensional alignment. |
| **3** | **Dimension Reduction** | **All Parameters** | MRL is applied. FC2, FC3, and FC4 are trained to produce smaller (1024, 512, 256) effective embeddings. |
| **4** | **Multimodal** | **Vision Encoder Only** | The text components are frozen. The Vision Encoder is trained to align images with the student's own text embeddings. |

---

## Citation

This code is an implementation of:

```bibtex
@article{zhang2024jasper,
  title={Jasper and Stella: Distillation of SOTA embedding models},
  author={Zhang, Dun and Li, Jiacheng and Zeng, Ziyang and Wang, Fulong},
  journal={arXiv preprint arXiv:2412.19048},
  year={2024}
}
```

*Note: This is an unofficial implementation created for research and verification purposes.*