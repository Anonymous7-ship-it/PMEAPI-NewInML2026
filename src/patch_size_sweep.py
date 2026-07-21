import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import src.config as cfg
from src.data_loader import get_partition, build_partition_indices
from src.model       import get_model, get_base_model
from src.attack      import train_patch, evaluate_attack
from src.classical   import detect_patch_batch
from src.feature_squeeze import load_threshold
from src.gradcam_defense  import GradCAM
from src.pipeline    import HybridDefense

SWEEP_SIZES = [44, 60, 75, 89, 110]   # pixels


SWEEP_SPEC = [("sweep_eval", 300)]     # 300 images per size


def _train_patch_for_size(px: int, model, train_loader) -> torch.Tensor:
    """Train a fresh patch at a given pixel size.

    Rotation augmentation (grid_sample) is disabled for the sweep because
    aten::grid_sampler_2d_backward is not implemented on MPS. The sweep
    measures ASR vs patch size under controlled conditions; rotation aug
    is not needed for this comparison.
    """
    original_size     = cfg.PATCH_SIZE_PX
    original_rotation = cfg.PATCH_ROTATION
    cfg.PATCH_SIZE_PX  = px
    cfg.PATCH_ROTATION = 0.0   # disable rotation to avoid grid_sampler MPS error

    import importlib, src.attack
    importlib.reload(src.attack)
    from src.attack import train_patch as _train

    patch = _train(model, train_loader, steps=cfg.PATCH_STEPS, save=False)
    cfg.PATCH_SIZE_PX  = original_size
    cfg.PATCH_ROTATION = original_rotation
    return patch


def _eval_stage1(images_clean, images_patched, n) -> dict:
    flags_patched, _, _ = detect_patch_batch(images_patched[:n], method='orb')
    flags_clean,   _, _ = detect_patch_batch(images_clean[:n],   method='orb')
    tp = sum(flags_patched)
    fp = sum(flags_clean)
    return {
        "stage1_detection": round(100 * tp / n, 2),
        "stage1_fpr":       round(100 * fp / n, 2),
    }


def _eval_full_pipeline(model, gradcam, images_clean, images_patched,
                        labels, fs_threshold, n) -> dict:
    defense = HybridDefense(
        model, gradcam, fs_threshold, kp_method='orb'
    )
    recovered = 0
    fp_count  = 0
    t0 = time.time()

    for i in range(n):
        res_p = defense.defend(images_patched[i],
                               true_label=labels[i].item())
        if res_p["correct"]:
            recovered += 1

        res_c = defense.defend(images_clean[i],
                               true_label=labels[i].item())
        if not res_c["correct"]:
            fp_count += 1

    elapsed = time.time() - t0
    return {
        "recovered_acc": round(100 * recovered / n, 2),
        "fpr":           round(100 * fp_count  / n, 2),
        "ms_per_image":  round(1000 * elapsed / (2 * n), 1),
    }


def run_sweep(train_loader, save: bool = True) -> dict:
    from src.data_loader import _PARTITION_SPEC
    extended_spec = _PARTITION_SPEC + SWEEP_SPEC
    sweep_parts   = build_partition_indices(spec=extended_spec)

    images_clean, labels = get_partition("sweep_eval", sweep_parts)
    images_clean = images_clean[:300]
    labels       = labels[:300]
    n            = images_clean.size(0)

    model   = get_model()
    gradcam = GradCAM(model, target_layer_name='layer4')
    fs_threshold = load_threshold()

    results = {}

    for px in SWEEP_SIZES:
        print(f"\n{'='*60}")
        print(f"Patch size: {px}x{px} "
              f"(~{100*(px/224)**2:.0f}% of image area)")
        print(f"{'='*60}")

        patch = _train_patch_for_size(px, model, train_loader)

        
        import importlib, src.attack as _atk_mod
        importlib.reload(_atk_mod)
        
        import sys
        sys.modules[__name__].__dict__.update(
            apply_patch=_atk_mod.apply_patch,
            evaluate_attack=_atk_mod.evaluate_attack,
        )

        # Attack success rate on undefended model
        asr_res = _atk_mod.evaluate_attack(model, patch, images_clean, labels)

        # Apply patch to get patched images for this size
        images_patched, _ = _atk_mod.apply_patch(
            images_clean.clone(), patch, augment=False
        )

        s1 = _eval_stage1(images_clean, images_patched, n)
        pipeline = _eval_full_pipeline(
            model, gradcam, images_clean, images_patched,
            labels, fs_threshold, n
        )

        results[px] = {
            "patch_px":      px,
            "area_pct":      round(100 * (px / 224) ** 2, 1),
            "asr":           asr_res["attack_sr"],
            **s1,
            **pipeline,
        }

        print(f"  ASR (undefended):     {results[px]['asr']:.1f}%")
        print(f"  Stage1 detection:     {results[px]['stage1_detection']:.1f}%")
        print(f"  Stage1 FPR:           {results[px]['stage1_fpr']:.1f}%")
        print(f"  Pipeline recovered:   {results[px]['recovered_acc']:.1f}%")
        print(f"  Pipeline FPR:         {results[px]['fpr']:.1f}%")
        print(f"  ms/image (defended):  {results[px]['ms_per_image']:.1f}")

    if save:
        out = os.path.join("results", "patch_size_sweep.pt")
        results['_valid'] = True   # marks this as a correctly-computed result
        torch.save(results, out)
        print("Sweep results saved.")
        results_clean = {k: v for k, v in results.items() if k != '_valid'}
        _plot_sweep(results_clean)

    return results


def print_sweep_table(results: dict):
    print("\n" + "=" * 88)
    print(f"{'Size':>6} {'Area':>6} {'ASR':>8} "
          f"{'S1 Det':>8} {'S1 FPR':>8} "
          f"{'Recov':>8} {'FPR':>8} {'ms/img':>8}")
    print("-" * 88)
    for px in sorted(k for k in results.keys() if isinstance(k, int)):
        r = results[px]
        mark = " *" if px == 89 else ""
        print(
            f"{px:>5}{mark:1s} {r['area_pct']:>5.1f}% "
            f"{r['asr']:>7.1f}% "
            f"{r['stage1_detection']:>7.1f}% "
            f"{r['stage1_fpr']:>7.1f}% "
            f"{r['recovered_acc']:>7.1f}% "
            f"{r['fpr']:>7.1f}% "
            f"{r['ms_per_image']:>7.1f}"
        )
    print("=" * 88)
    print("* Primary evaluation patch size")


def _plot_sweep(results: dict):
    sizes    = sorted(k for k in results.keys() if isinstance(k, int))
    asr      = [results[px]["asr"]               for px in sizes]
    s1_det   = [results[px]["stage1_detection"]  for px in sizes]
    pipeline = [results[px]["recovered_acc"]      for px in sizes]
    fpr      = [results[px]["fpr"]               for px in sizes]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(sizes, asr,      "o--", color="#888780", label="Undefended ASR")
    axes[0].plot(sizes, s1_det,   "s-",  color="#378ADD", label="Stage 1 detection")
    axes[0].plot(sizes, pipeline, "^-",  color="#1D9E75", label="Hybrid recovered acc")
    axes[0].axvline(89, color="gray", linestyle=":", alpha=0.7, label="Primary patch (89px)")
    axes[0].set_xlabel("Patch size (px)")
    axes[0].set_ylabel("Percentage (%)")
    axes[0].set_title("Effectiveness vs patch size")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)
    axes[0].set_xticks(sizes)

    axes[1].plot(sizes, fpr, "o-", color="#D85A30", label="Pipeline FPR")
    axes[1].axvline(89, color="gray", linestyle=":", alpha=0.7)
    axes[1].set_xlabel("Patch size (px)")
    axes[1].set_ylabel("FPR (%)")
    axes[1].set_title("False positive rate vs patch size")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)
    axes[1].set_xticks(sizes)

    plt.tight_layout()
    out = os.path.join("results", "plots", "patch_size_sweep.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print("Sweep plot saved.")