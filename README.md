# PMEAPI: Probabilistic Multi-Evidence Adversarial Patch Inference

Code and experiments for the submission *"PMEAPI: Probabilistic Multi-Evidence Adversarial Patch
Inference"* (submitted to AISec 2026, anonymous review).

This repository contains the full pipeline used to produce every table and figure in the paper:
a three-stage hybrid defense against adversarial patch attacks (classical keypoint detection →
feature squeezing → GradCAM-guided masking), a probabilistic fusion variant (**PMEAPI**), an
adaptive (defense-aware) attack evaluation, and comparisons against JPEG compression,
feature-squeezing-only, GradCAM-only, and PatchCleanser baselines on ImageNette.

> **Note on anonymity:** this repository has been prepared for anonymous review. Do not include
> any identifying information (names, institutions, emails, funding acknowledgements, local file
> paths, git history from a non-anonymous remote) in any file added here.

## Repository structure

```
.
├── notebooks/                     # Numbered so review follows the pipeline in order
│   ├── 01_main_pipeline.ipynb     # Patch training, Stage 1-3 hybrid defense, full 7-method evaluation (Table II)
│   ├── 02_adaptive_attack.ipynb   # Defense-aware adaptive patch, oblivious vs. adaptive comparison (Sec. IV-G)
│   ├── 03_sweep_patchcleanser.ipynb  # Patch-size sweep + PatchCleanser (Xiang et al., USENIX Sec'22) baseline
│   ├── 04_pmeapi.ipynb            # Probabilistic (Beta-calibrated) fusion defense: calibration, threshold tuning
│   └── 05_final_figures.ipynb     # Regenerates every camera-ready figure from saved results only
├── src/
│   ├── config.py                  # All hyperparameters, paths, dataset constants
│   ├── data_loader.py             # ImageNette download + disjoint partitioning across pipeline stages
│   ├── model.py                   # ResNet-50 -> ImageNette label wrapper
│   ├── attack.py                  # Adversarial patch training + application
│   ├── adaptive_attack.py         # Defense-aware patch (evades classical + feature-squeeze stages)
│   ├── adaptive_attack_pmeapi.py  # Defense-aware patch targeting PMEAPI's fused posterior directly
│   ├── classical.py               # Stage 1: ORB/SIFT/AKAZE keypoint-density patch localization
│   ├── feature_squeeze.py         # Stage 2: bit-depth + median-filter squeezing detector
│   ├── gradcam_defense.py         # Stage 3: GradCAM localization + masking
│   ├── pipeline.py                # HybridDefense: chains Stages 1-3
│   ├── probabilistic_fusion.py    # PMEAPI: Beta-calibrated likelihood-ratio fusion of all three sensors
│   ├── patchcleanser_baseline.py  # PatchCleanser double-masking baseline
│   ├── patch_size_sweep.py        # Effectiveness vs. patch size (44-110px) sweep
│   ├── evaluate.py                # Bootstrap CIs, comparison tables, all summary plots
│   ├── stats.py                   # Bootstrap / Wilson CI helpers, significance tests
│   ├── generate_final_figures.py  # Standalone script: PMEAPI threshold-analysis figure from saved calibration
│   └── generate_supporting_figures.py  # Standalone script: sensor calibration + Table II figures
└── results/                       # Populated by running the notebooks (not tracked in git; see .gitignore)
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Tested on Python 3.10. Runs on CUDA, Apple Silicon (MPS), or CPU — device is auto-selected in
`src/config.py`. Note: patch training disables rotation augmentation when running on MPS, since
`grid_sampler_2d_backward` is not implemented for that backend (see comments in
`src/adaptive_attack.py` / `src/patch_size_sweep.py`).

## Data

Experiments use [ImageNette](https://github.com/fastai/imagenette) (a 10-class ImageNet subset).
It is downloaded automatically to `data/` on first run of `src/data_loader.get_loaders()` — no
manual steps required. The dataset is not included in this repository.

## Reproducing the paper's results

Run the notebooks in numeric order:

1. **`01_main_pipeline.ipynb`** — trains the base 89px patch, builds the disjoint data partitions
   used throughout (`stage1_bench`, `fs_tuning`, `fs_dev`, `stage3_dev`, `final_eval`), and produces
   the main 7-method comparison (Table II): no-defense, JPEG, feature-squeezing, GradCAM-only,
   hybrid pipeline, PMEAPI, PatchCleanser.
2. **`02_adaptive_attack.ipynb`** — trains a defense-aware adaptive patch against the hybrid
   pipeline and reports the oblivious-vs-adaptive degradation with bootstrap CIs (Sec. IV-G).
3. **`03_pmeapi.ipynb`** — fits per-sensor Beta likelihoods (ORB, feature-squeeze, GradCAM
   concentration) on held-out calibration data and tunes the fused decision threshold for a
   target FPR.
4. **`04_sweep_patchcleanser.ipynb`** — sweeps patch size (44–110px) against the hybrid pipeline
   and runs the PatchCleanser certified-defense baseline for comparison.
5. **`05_final_figures.ipynb`** — regenerates every camera-ready figure from the `results/*.pt`
   files saved by the notebooks above (no retraining or re-evaluation required).

All notebooks are checkpoint-safe: long-running cells (patch training, full-pipeline evaluation)
save intermediate state to `results/checkpoints/` and skip recomputation if a valid checkpoint is
already present, so notebooks can be interrupted and resumed.

## Results (Table II summary, 1,000-image final_eval, 95% bootstrap CI)

| Method             | Clean acc | Attack SR | Recovered acc | FPR |
|--------------------|-----------|-----------|----------------|-----|
| No defense         | 99.8%     | 84.3%     | 15.7%          | 0.0% |
| JPEG compression   | 99.8%     | 46.1%     | 53.9%          | 0.2% |
| Feature squeezing  | 90.6%     | 21.2%     | 78.8%          | 9.4% |
| GradCAM only       | 96.5%     | 2.3%      | 97.7%          | 3.5% |
| PatchCleanser      | 99.6%     | 34.9%     | 65.1%          | 0.4% |
| Hybrid pipeline    | 99.5%     | 15.3%     | 84.7%          | 0.5% |
| **PMEAPI (ours)**  | 99.8%     | 7.4%      | 92.6%          | 0.2% |

Full CIs and per-figure provenance are in `notebooks/05_final_figures.ipynb`.

## Use of Generative AI

The author used Claude (Anthropic) as a coding and writing assis-
tant during preparation of this research. All experimental results,
analysis, and scientific claims are the author’s own. Use of AI tools
complies with ACM policies


