import os
import itertools
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.config  import DEVICE, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, PLOTS_DIR
from src.model   import get_model


def _generate_mask_set(image_size: int, k: int, patch_size: int):
    """Generate the complete set of single masks for a k-partition grid.
    
    Divides the image into k strips along each axis, creating k^2 cells.
    Each mask zeros out one cell.  Returns list of (row_start, col_start,
    row_end, col_end) tuples in pixel coordinates.
    
    For PatchCleanser the mask size must be >= patch size to guarantee
    that every possible patch placement is covered by at least one mask.
    """
    strip_h = image_size // k
    strip_w = image_size // k

    masks = []
    for i in range(k):
        for j in range(k):
            r1 = i * strip_h
            c1 = j * strip_w
            r2 = min(r1 + strip_h, image_size)
            c2 = min(c1 + strip_w, image_size)
            # Expand mask to at least cover patch_size
            r2 = min(r1 + max(strip_h, patch_size), image_size)
            c2 = min(c1 + max(strip_w, patch_size), image_size)
            masks.append((r1, c1, r2, c2))
    return masks


def _apply_mask(image: torch.Tensor, mask: tuple) -> torch.Tensor:
    """Zero out a rectangular region (in normalised space, 0 = ImageNet mean)."""
    masked = image.clone()
    r1, c1, r2, c2 = mask
    masked[:, r1:r2, c1:c2] = 0.0  # 0 in normalised space = ImageNet mean
    return masked


@torch.no_grad()
def _predict_masked(model, image: torch.Tensor, mask) -> int:
    """mask is either a single (r1,c1,r2,c2) tuple or a pair of such tuples."""
    if isinstance(mask[0], tuple):
        # double mask: apply both regions
        masked = image.clone()
        for m in mask:
            r1, c1, r2, c2 = m
            masked[:, r1:r2, c1:c2] = 0.0
    else:
        masked = _apply_mask(image, mask)
    logits = model(masked.unsqueeze(0).to(DEVICE))
    return logits.argmax(dim=1).item()


@torch.no_grad()
def patchcleanser_predict(
    model,
    image: torch.Tensor,
    single_masks: list,
    double_masks: list,
) -> tuple[int, bool]:
    """Run PatchCleanser double-masking inference on one image.
    
    Returns (prediction, is_certified):
      - is_certified=True means all single-masked predictions agree
        (provably robust under the mask-set guarantee).
      - is_certified=False means we fell through to majority vote on
        double-masked predictions (weaker guarantee, but still often
        correct empirically).
    """
    single_preds = [_predict_masked(model, image, m) for m in single_masks]

    if len(set(single_preds)) == 1:
        return single_preds[0], True

    double_preds = [
        _predict_masked(model, image, m) for m in double_masks
    ]

    from collections import Counter
    vote = Counter(double_preds).most_common(1)[0][0]
    return vote, False


def evaluate_patchcleanser(
    model,
    images_clean:   torch.Tensor,
    images_patched: torch.Tensor,
    labels:         torch.Tensor,
    patch_size_px:  int = 89,
    k:              int = 2,
) -> dict:
    """Evaluate PatchCleanser double-masking on clean and patched images.
    
    k=2 -> 4 single masks, 6 double masks (10 forward passes per image)
    k=3 -> 9 single masks, 36 double masks (45 forward passes per image)
    
    For a 89px patch (~16% of image), k=2 is the recommended setting.
    For smaller patches, k=3 or k=4 gives stronger certified guarantees.
    """
    single_masks = _generate_mask_set(IMAGE_SIZE, k, patch_size_px)
    double_masks = list(itertools.combinations(single_masks, 2))

    print(
        f"\nPatchCleanser evaluation: k={k}, "
        f"{len(single_masks)} single masks, {len(double_masks)} double masks, "
        f"patch_size={patch_size_px}px"
    )

    N = labels.size(0)
    labels_cpu = labels.cpu()

    clean_correct    = 0
    patched_correct  = 0
    certified_clean  = 0
    certified_patched = 0

    import torch as _torch
    for i in tqdm(range(N), desc="  PatchCleanser (clean)"):
        pred, cert = patchcleanser_predict(
            model, images_clean[i], single_masks, double_masks
        )
        if pred == labels_cpu[i].item():
            clean_correct += 1
        if cert:
            certified_clean += 1
        if DEVICE.type == "mps":
            _torch.mps.empty_cache()

    for i in tqdm(range(N), desc="  PatchCleanser (patched)"):
        pred, cert = patchcleanser_predict(
            model, images_patched[i], single_masks, double_masks
        )
        if pred == labels_cpu[i].item():
            patched_correct += 1
        if cert:
            certified_patched += 1
        if DEVICE.type == "mps":
            _torch.mps.empty_cache()

    result = {
        "clean_acc":           round(100 * clean_correct   / N, 2),
        "recovered_acc":       round(100 * patched_correct / N, 2),
        "fpr":                 round(100 * (N - clean_correct) / N, 2),
        "certified_clean_pct": round(100 * certified_clean    / N, 2),
        "certified_patched_pct": round(100 * certified_patched / N, 2),
        "k":                   k,
        "n_images":            N,
        "n_single_masks":      len(single_masks),
        "n_double_masks":      len(double_masks),
        "n_forward_passes_per_image": len(single_masks) + len(double_masks),
    }

    print(f"\nPatchCleanser (k={k}) results:")
    print(f"  Clean accuracy:          {result['clean_acc']:.1f}%")
    print(f"  Recovered accuracy:      {result['recovered_acc']:.1f}%")
    print(f"  False positive rate:     {result['fpr']:.1f}%")
    print(f"  Certified (clean):       {result['certified_clean_pct']:.1f}%")
    print(f"  Certified (patched):     {result['certified_patched_pct']:.1f}%")
    print(f"  Forward passes/image:    {result['n_forward_passes_per_image']}")

    return result


def add_patchcleanser_to_table(
    existing_results: dict,
    pc_result: dict,
) -> dict:
    """Add PatchCleanser as a new row in the results dict so it appears
    in Table II alongside the other five methods.
    
    Note: PatchCleanser's 'FPR' here is the fraction of clean images
    misclassified after double masking -- this is NOT the same as the
    certified FPR from their paper (which is 0 by construction for
    certified images).  We report it as empirical FPR under the same
    metric definition as our other methods, and note this in the caption.
    """
    from src.evaluate import bootstrap_confidence_interval
    N = pc_result.get("n_images", 1000)  # use actual eval N for correct CIs

    def _bci(successes_pct):
        successes = int(round(successes_pct * N / 100))
        _, lo, hi = bootstrap_confidence_interval(successes, N, n_bootstrap=1000)
        return round(lo, 1), round(hi, 1)

    clean_lo,   clean_hi   = _bci(pc_result["clean_acc"])
    rec_lo,     rec_hi     = _bci(pc_result["recovered_acc"])
    atk_lo,     atk_hi     = _bci(100 - pc_result["recovered_acc"])
    fpr_lo,     fpr_hi     = _bci(pc_result["fpr"])

    existing_results["patchcleanser"] = {
        "clean_acc":     pc_result["clean_acc"],
        "clean_ci_low":  clean_lo,
        "clean_ci_high": clean_hi,
        "recovered_acc": pc_result["recovered_acc"],
        "rec_ci_low":    rec_lo,
        "rec_ci_high":   rec_hi,
        "attack_sr":     100 - pc_result["recovered_acc"],
        "attack_ci_low":  atk_lo,
        "attack_ci_high": atk_hi,
        "fpr":           pc_result["fpr"],
        "fpr_ci_low":    fpr_lo,
        "fpr_ci_high":   fpr_hi,
        "note": (
            f"PatchCleanser double-masking (k={pc_result['k']}, "
            f"{pc_result['n_forward_passes_per_image']} passes/img, n={N}). "
            f"Empirical (not certified) FPR reported for comparability. "
            f"For certified robustness numbers see Xiang et al. 2022 Table 2."
        ),
    }
    return existing_results


PATCHCLEANSER_LITERATURE = {
    "dataset":        "ImageNette",
    "model":          "ResNet-50 (smoothed)",
    "clean_acc":      99.1,   # from Table 2, k=6 setting
    "certified_acc":  49.8,   # certified robust accuracy, 2% patch
    "patch_size_pct": 2.0,    # their evaluation uses 2% patch area
    "source":         "Xiang et al., USENIX Security 2022, Table 2",
    "note": (
        "Their certified accuracy is for 2% patch area directly "
        "comparable to our 4% (44px) setting, not our primary 16% (89px) "
        "setting. Ours is a harder attack; certified defenses degrade "
        "sharply as patch size grows."
    ),
}