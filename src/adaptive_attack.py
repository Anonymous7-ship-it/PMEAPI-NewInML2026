import math
import os
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image, to_tensor
from PIL import ImageFilter
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.config import (
    DEVICE, PATCH_SIZE_PX, TARGET_CLASS, PATCH_LR,
    PATCH_BATCH, PATCH_ROTATION, PLOTS_DIR, IMAGENET_MEAN, IMAGENET_STD,
    IMAGENETTE_TO_IMAGENET,
)
from src.attack import apply_patch, random_transform_patch
from src.feature_squeeze import squeeze_bit_depth, squeeze_median

TARGET_IMAGENET = IMAGENETTE_TO_IMAGENET[TARGET_CLASS]

LAMBDA_KP = 0.5    # total-variation weight  (keypoint evasion)
LAMBDA_FS = 0.3    # squeeze-stability weight (feature-squeeze evasion)

ADAPTIVE_STEPS   = 1000

from src.config import PATCH_DIR
ADAPTIVE_PATCH_PATH = os.path.join(PATCH_DIR, "patch_adaptive_89px.pt")


def _tv_loss(patch: torch.Tensor) -> torch.Tensor:
    """Total variation of the patch -- proxy for boundary sharpness.
    Minimising TV encourages smooth colour gradients, which produce
    fewer FAST/ORB keypoints at the patch edge.
    """
    diff_h = (patch[:, 1:, :] - patch[:, :-1, :]).abs().mean()
    diff_w = (patch[:, :, 1:] - patch[:, :, :-1]).abs().mean()
    return diff_h + diff_w


def _squeeze_stability_loss(
    model,
    patched_images: torch.Tensor,
) -> torch.Tensor:
    """Penalise L1 distance between the model's softmax output on the
    patched image and on its bit-depth-reduced version.  Minimising this
    encourages the patch to survive bit-depth reduction without causing
    prediction instability -- making Stage 2's score drop below theta.
    """
    probs_orig = F.softmax(model(patched_images), dim=1)
    sq = torch.stack([squeeze_bit_depth(patched_images[i])
                      for i in range(patched_images.size(0))])
    probs_sq = F.softmax(model(sq.to(DEVICE)), dim=1)
    return (probs_orig - probs_sq).abs().sum(dim=1).mean()


def train_adaptive_patch(
    model,
    data_loader,
    steps: int = ADAPTIVE_STEPS,
    lambda_kp: float = LAMBDA_KP,
    lambda_fs: float = LAMBDA_FS,
    save: bool = True,
) -> torch.Tensor:
    """Train an adaptive patch that explicitly tries to evade all three
    stages of the hybrid defense while still causing misclassification.
    """
    print(
        f"\nTraining ADAPTIVE patch: {steps} steps, "
        f"lambda_kp={lambda_kp}, lambda_fs={lambda_fs}"
    )
    import src.config as cfg
    _original_rotation = cfg.PATCH_ROTATION
    cfg.PATCH_ROTATION = 0.0   

    patch = torch.rand(
        3, PATCH_SIZE_PX, PATCH_SIZE_PX,
        requires_grad=True, device=DEVICE
    )
    optimizer = torch.optim.Adam([patch], lr=PATCH_LR)

    from src.model import get_base_model
    base_model = get_base_model()
    base_model.eval()

    data_iter = iter(data_loader)
    target_t  = torch.tensor([TARGET_IMAGENET] * PATCH_BATCH, device=DEVICE)

    losses_total = []
    losses_ce    = []
    losses_kp    = []
    losses_fs    = []

    for step in tqdm(range(steps), desc="Adaptive patch training"):
        try:
            images, _ = next(data_iter)
        except StopIteration:
            data_iter  = iter(data_loader)
            images, _  = next(data_iter)

        images = images[:PATCH_BATCH].to(DEVICE)
        optimizer.zero_grad()

        p_clamped = patch.clamp(0, 1)
        patched, _ = apply_patch(images, p_clamped, augment=True, max_angle=0.0)

        # 1. Classification loss 
        logits = base_model(patched)
        loss_ce = F.cross_entropy(logits, target_t[:images.size(0)])

        # 2. Keypoint evasion: minimise total variation of the patch
        loss_kp = _tv_loss(p_clamped)

        # 3. Feature-squeeze evasion: minimise prediction shift under squeezing
        loss_fs = _squeeze_stability_loss(base_model, patched)

        loss = loss_ce + lambda_kp * loss_kp + lambda_fs * loss_fs
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            patch.clamp_(0, 1)

        losses_total.append(loss.item())
        losses_ce.append(loss_ce.item())
        losses_kp.append(loss_kp.item())
        losses_fs.append(loss_fs.item())

        if (step + 1) % 100 == 0:
            print(
                f"  step {step+1:4d}/{steps}  "
                f"total={loss.item():.4f}  "
                f"ce={loss_ce.item():.4f}  "
                f"kp={loss_kp.item():.4f}  "
                f"fs={loss_fs.item():.4f}"
            )

    final_patch = patch.detach().clamp(0, 1)

    if save:
        torch.save(final_patch, ADAPTIVE_PATCH_PATH)
        print(f"Adaptive patch saved")
        _plot_adaptive_losses(losses_ce, losses_kp, losses_fs)

    cfg.PATCH_ROTATION = _original_rotation
    return final_patch


def evaluate_adaptive(
    model,
    hybrid_defense,
    patch: torch.Tensor,
    images: torch.Tensor,
    labels: torch.Tensor,
    label: str = "Adaptive patch",
) -> dict:
    """Evaluate attack success rate and full-pipeline defense recovery
    for the adaptive patch, returning a results dict comparable to
    run_full_evaluation's format so it can be added as a table row.
    """
    from src.model import get_base_model
    base_model = get_base_model()
    base_model.eval()

    N = images.size(0)
    patched_images, _ = apply_patch(images.clone(), patch, augment=False)

    # Undefended attack success rate
    with torch.no_grad():
        clean_preds   = model(images.to(DEVICE)).argmax(1).cpu()
        patched_preds = base_model(patched_images.to(DEVICE)).argmax(1).cpu()

    correctly_clean = (clean_preds == labels.cpu())
    now_wrong = (patched_preds != labels.cpu())
    asr = (correctly_clean & now_wrong).float().sum().item() / max(correctly_clean.sum().item(), 1)

    # Hybrid defense recovery on the adaptive patch
    recovered = 0
    for i in tqdm(range(N), desc=f"  Hybrid vs {label}"):
        res = hybrid_defense.defend(
            patched_images[i], true_label=labels[i].item()
        )
        if res["correct"]:
            recovered += 1

    recovered_acc = 100 * recovered / N

    result = {
        "asr_undefended":  round(100 * asr, 2),
        "recovered_acc":   round(recovered_acc, 2),
        "n_images":        N,
        "label":           label,
    }
    print(
        f"\n{label}:\n"
        f"  Undefended ASR:   {result['asr_undefended']:.1f}%\n"
        f"  Hybrid recovery:  {result['recovered_acc']:.1f}%"
    )
    return result


def compare_standard_vs_adaptive(
    model,
    hybrid_defense,
    patch_standard: torch.Tensor,
    patch_adaptive: torch.Tensor,
    images: torch.Tensor,
    labels: torch.Tensor,
) -> dict:
    """Run both patches through the hybrid defense and print a comparison
    table suitable for inclusion in Section IV-G of the paper.
    """
    res_std = evaluate_adaptive(
        model, hybrid_defense, patch_standard, images, labels,
        label="Standard patch (oblivious)"
    )
    res_adp = evaluate_adaptive(
        model, hybrid_defense, patch_adaptive, images, labels,
        label="Adaptive patch (defense-aware)"
    )

    print("\n" + "=" * 65)
    print(f"{'':30s} {'Oblivious':>15} {'Adaptive':>15}")
    print("-" * 65)
    print(f"{'Undefended ASR':30s} {res_std['asr_undefended']:>14.1f}% "
          f"{res_adp['asr_undefended']:>14.1f}%")
    print(f"{'Hybrid recovered acc':30s} {res_std['recovered_acc']:>14.1f}% "
          f"{res_adp['recovered_acc']:>14.1f}%")
    print(f"{'Defense degradation':30s} "
          f"{'—':>15} "
          f"{res_std['recovered_acc'] - res_adp['recovered_acc']:>13.1f}pp")
    print("=" * 65)
    print("\nInterpretation:")
    delta = res_std['recovered_acc'] - res_adp['recovered_acc']
    if delta < 5:
        print(f"  Delta = {delta:.1f}pp -- the hybrid defense is ROBUST to this "
              f"adaptive attack. Recovered accuracy drops only {delta:.1f} "
              f"percentage points when the attacker knows the full defense. ")
    elif delta < 15:
        print(f"  Delta = {delta:.1f}pp -- the adaptive attack causes MODERATE "
              f"degradation. Recovery drops from {res_std['recovered_acc']:.1f}% "
              f"to {res_adp['recovered_acc']:.1f}%")
    else:
        print(f"  Delta = {delta:.1f}pp -- the adaptive attack causes SUBSTANTIAL "
              f"degradation. "
              f"adaptive-robust Stage 1 detectors.")

    return {"standard": res_std, "adaptive": res_adp, "delta_pp": delta}


def _plot_adaptive_losses(ce, kp, fs):
    path = os.path.join(PLOTS_DIR, "adaptive_patch_losses.png")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, vals, title, color in zip(
        axes,
        [ce, kp, fs],
        ["Classification loss (CE)", "TV loss (keypoint evasion)",
         "Squeeze-stability loss (FS evasion)"],
        ["#D85A30", "#1D9E75", "#534AB7"]
    ):
        ax.plot(vals, linewidth=0.8, color=color)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Step")
        ax.grid(alpha=0.3)
    plt.suptitle("Adaptive patch training losses", fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Adaptive loss curves saved")