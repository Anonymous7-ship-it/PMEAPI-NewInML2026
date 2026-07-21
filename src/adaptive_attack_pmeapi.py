
import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.config import (
    DEVICE, PATCH_SIZE_PX, TARGET_CLASS, PATCH_LR,
    PATCH_BATCH, PLOTS_DIR, IMAGENETTE_TO_IMAGENET,
    PATCH_DIR   # <-- new import
)
from src.attack import apply_patch
from src.feature_squeeze import squeeze_bit_depth
from src.probabilistic_fusion import load_calibration

TARGET_IMAGENET = IMAGENETTE_TO_IMAGENET[TARGET_CLASS]

ADAPTIVE_PMEAPI_STEPS = 1000
ADAPTIVE_PMEAPI_PATCH_PATH = os.path.join(PATCH_DIR, "patch_adaptive_pmeapi_89px.pt")

# Same TV/squeeze-stability weights as the original adaptive_attack.py,
# for direct comparability of the ORB/FS regularisation terms.
LAMBDA_KP   = 0.5   # TV loss weight (ORB evasion proxy -- NOT part of the posterior term)
LAMBDA_FS   = 0.3   # squeeze-stability weight (own regulariser, same as original attack)
LAMBDA_GC   = 0.5   # GradCAM concentration weight (own regulariser, targets Stage 3 directly)
LAMBDA_POST = 1.0   # direct FS+GC partial-posterior weight (the NEW component vs. the original attack)


def _tv_loss(patch: torch.Tensor) -> torch.Tensor:
    diff_h = (patch[:, 1:, :] - patch[:, :-1, :]).abs().mean()
    diff_w = (patch[:, :, 1:] - patch[:, :, :-1]).abs().mean()
    return diff_h + diff_w


def _differentiable_fs_score(model, patched_images: torch.Tensor) -> torch.Tensor:
   
    probs_orig = F.softmax(model(patched_images), dim=1)
    sq = torch.stack([squeeze_bit_depth(patched_images[i]) for i in range(patched_images.size(0))])
    probs_sq = F.softmax(model(sq.to(DEVICE)), dim=1)
    l1 = (probs_orig - probs_sq).abs().sum(dim=1)  # (B,)
    return l1


def _differentiable_gradcam_concentration(
    model, images: torch.Tensor, target_layer_name: str = "layer4", top_frac: float = 0.15,
) -> torch.Tensor:
    
    base = model.base if hasattr(model, "base") else model
    target_layer = dict(base.named_modules())[target_layer_name]

    activations = {}

    def _fwd_hook(module, inp, out):
        activations["value"] = out

    handle = target_layer.register_forward_hook(_fwd_hook)
    logits = model(images)
    handle.remove()

    pred_class = logits.argmax(dim=1)
    target_logits = logits.gather(1, pred_class.unsqueeze(1)).squeeze(1)

    grads = torch.autograd.grad(
        outputs=target_logits.sum(),
        inputs=activations["value"],
        create_graph=True,   # keep this in the graph for the OUTER backward() call
        retain_graph=True,
    )[0]

    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * activations["value"]).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=images.shape[-2:], mode="bilinear", align_corners=False)

    B = cam.size(0)
    cam_flat = cam.view(B, -1)
    cam_max = cam_flat.max(dim=1, keepdim=True).values.clamp_min(1e-8)
    cam_norm = cam_flat / cam_max

    k = max(1, int(top_frac * cam_flat.size(1)))
    topk_vals, _ = torch.topk(cam_norm, k, dim=1)
    concentration = topk_vals.mean(dim=1)
    return concentration.clamp(0.0, 1.0)


def _torch_log_beta_pdf(x: torch.Tensor, a: float, b: float) -> torch.Tensor:
    
    eps = 1e-6
    x = x.clamp(eps, 1 - eps)
    log_norm = torch.lgamma(torch.tensor(a + b, device=x.device, dtype=x.dtype)) \
        - torch.lgamma(torch.tensor(a, device=x.device, dtype=x.dtype)) \
        - torch.lgamma(torch.tensor(b, device=x.device, dtype=x.dtype))
    return (a - 1) * torch.log(x) + (b - 1) * torch.log(1 - x) + log_norm


def _differentiable_llr(x: torch.Tensor, a0: float, b0: float, a1: float, b1: float) -> torch.Tensor:
    log_p1 = _torch_log_beta_pdf(x, a1, b1)
    log_p0 = _torch_log_beta_pdf(x, a0, b0)
    return torch.clamp(log_p1 - log_p0, -20, 20)


def _fs_gc_partial_posterior(fs_score_raw: torch.Tensor, gc_score: torch.Tensor, params: dict) -> torch.Tensor:
    
    fs_norm = (fs_score_raw / 1.75).clamp(0.0, 1.0)  # same normalisation as _fs_score_normalised

    llr_fs = _differentiable_llr(
        fs_norm, params["fs"]["alpha_0"], params["fs"]["beta_0"],
        params["fs"]["alpha_1"], params["fs"]["beta_1"],
    )
    llr_gc = _differentiable_llr(
        gc_score, params["gc"]["alpha_0"], params["gc"]["beta_0"],
        params["gc"]["alpha_1"], params["gc"]["beta_1"],
    )
    log_odds = llr_fs + llr_gc  # prior_log_odds omitted (0.0 for equal priors, per config)
    return torch.sigmoid(log_odds)


def train_adaptive_patch_pmeapi(
    model, data_loader,
    steps: int = ADAPTIVE_PMEAPI_STEPS,
    lambda_kp: float = LAMBDA_KP,
    lambda_fs: float = LAMBDA_FS,
    lambda_gc: float = LAMBDA_GC,
    lambda_post: float = LAMBDA_POST,
    save: bool = True,
) -> torch.Tensor:
    """Train a patch that directly attacks PMEAPI's fused posterior
    (FS+GC partial surrogate) in addition to the same TV/squeeze-stability
    regularisers the original adaptive attack used against the binary
    cascade. See module docstring for the differentiability caveats.
    """
    print(
        f"\nTraining PMEAPI-targeted adaptive patch: {steps} steps, "
        f"lambda_kp={lambda_kp}, lambda_fs={lambda_fs}, lambda_gc={lambda_gc}, "
        f"lambda_post={lambda_post}"
    )

    params = load_calibration()
    if "_raw_scores" in params:
        params = {k: v for k, v in params.items() if k != "_raw_scores"}

    patch = torch.rand(3, PATCH_SIZE_PX, PATCH_SIZE_PX, requires_grad=True, device=DEVICE)
    optimizer = torch.optim.Adam([patch], lr=PATCH_LR)

    from src.model import get_base_model
    base_model = get_base_model()
    base_model.eval()

    data_iter = iter(data_loader)
    target_t = torch.tensor([TARGET_IMAGENET] * PATCH_BATCH, device=DEVICE)

    losses_total, losses_ce, losses_kp, losses_fs, losses_gc, losses_post = [], [], [], [], [], []

    for step in tqdm(range(steps), desc="PMEAPI-adaptive patch training"):
        try:
            images, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            images, _ = next(data_iter)

        images = images[:PATCH_BATCH].to(DEVICE)
        optimizer.zero_grad()

        p_clamped = patch.clamp(0, 1)
        # max_angle=0.0 explicitly bypasses F.affine_grid/F.grid_sample, whose
        # backward pass (grid_sampler_2d_backward) is not implemented on MPS --
        # same fix already applied to adaptive_attack.py's training loop.
        patched, _ = apply_patch(images, p_clamped, augment=True, max_angle=0.0)

        # 1. Classification loss (must fool the model)
        logits = base_model(patched)
        loss_ce = F.cross_entropy(logits, target_t[:images.size(0)])

        # 2. ORB evasion proxy (TV) -- NOT part of the posterior surrogate
        loss_kp = _tv_loss(p_clamped)

        # 3. FS regulariser (own term, same as original attack)
        fs_raw = _differentiable_fs_score(base_model, patched)   # (B,)
        loss_fs = fs_raw.mean()

        # 4. GC regulariser (own term, new vs. original attack)
        gc_score = _differentiable_gradcam_concentration(base_model, patched)  # (B,)
        loss_gc = gc_score.mean()

        # 5. Direct FS+GC partial posterior (the actual PMEAPI-targeting term)
        posterior_fs_gc = _fs_gc_partial_posterior(fs_raw, gc_score, params)
        loss_post = posterior_fs_gc.mean()

        loss = (
            loss_ce
            + lambda_kp * loss_kp
            + lambda_fs * loss_fs
            + lambda_gc * loss_gc
            + lambda_post * loss_post
        )
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            patch.clamp_(0, 1)

        losses_total.append(loss.item())
        losses_ce.append(loss_ce.item())
        losses_kp.append(loss_kp.item())
        losses_fs.append(loss_fs.item())
        losses_gc.append(loss_gc.item())
        losses_post.append(loss_post.item())

        if (step + 1) % 100 == 0:
            print(
                f"  step {step+1:4d}/{steps}  total={loss.item():.4f}  "
                f"ce={loss_ce.item():.4f}  kp={loss_kp.item():.4f}  "
                f"fs={loss_fs.item():.4f}  gc={loss_gc.item():.4f}  "
                f"posterior(fs+gc)={loss_post.item():.4f}"
            )

    final_patch = patch.detach().clamp(0, 1)

    if save:
        torch.save(final_patch, ADAPTIVE_PMEAPI_PATCH_PATH)
        print(f"PMEAPI-adaptive patch saved")
        _plot_losses(losses_ce, losses_kp, losses_fs, losses_gc, losses_post)

    return final_patch


def evaluate_adaptive_pmeapi(
    pmeapi_defense,
    patch: torch.Tensor,
    images: torch.Tensor,
    labels: torch.Tensor,
    label: str = "PMEAPI-adaptive patch",
) -> dict:
    """Evaluate the PMEAPI-targeted patch against the REAL, exact,
    non-differentiable PMEAPiDefense end-to-end -- the same code path
    used for every other number in the paper. This is the authoritative
    result; the training loss above is only a means to find the patch.
    """
    from src.model import get_base_model
    base_model = get_base_model()
    base_model.eval()

    N = images.size(0)
    patched_images, _ = apply_patch(images.clone(), patch, augment=False)

    with torch.no_grad():
        clean_preds = base_model(images.to(DEVICE)).argmax(1).cpu()
        patched_preds = base_model(patched_images.to(DEVICE)).argmax(1).cpu()

    correctly_clean = (clean_preds == labels.cpu())
    now_wrong = (patched_preds != labels.cpu())
    asr = (correctly_clean & now_wrong).float().sum().item() / max(correctly_clean.sum().item(), 1)

    recovered = 0
    posteriors_observed = []
    for i in tqdm(range(N), desc=f"  PMEAPI vs {label}"):
        res = pmeapi_defense.defend(patched_images[i], true_label=labels[i].item())
        if res["correct"]:
            recovered += 1
        posteriors_observed.append(res["patch_prob"])

    recovered_acc = 100 * recovered / N

    result = {
        "asr_undefended": round(100 * asr, 2),
        "recovered_acc": round(recovered_acc, 2),
        "mean_posterior": round(float(np.mean(posteriors_observed)), 4),
        "n_images": N,
        "label": label,
    }
    print(
        f"\n{label}:\n"
        f"  Undefended ASR:     {result['asr_undefended']:.1f}%\n"
        f"  PMEAPI recovery:    {result['recovered_acc']:.1f}%\n"
        f"  Mean real posterior on patched images: {result['mean_posterior']:.3f}"
    )
    return result


def compare_standard_vs_adaptive_pmeapi(
    pmeapi_defense,
    patch_standard: torch.Tensor,
    patch_adaptive_pmeapi: torch.Tensor,
    images: torch.Tensor,
    labels: torch.Tensor,
) -> dict:
    """Head-to-head comparison table, mirroring adaptive_attack.py's
    compare_standard_vs_adaptive() but against the real PMEAPiDefense."""
    res_std = evaluate_adaptive_pmeapi(
        pmeapi_defense, patch_standard, images, labels,
        label="Standard patch (oblivious) vs PMEAPI",
    )
    res_adp = evaluate_adaptive_pmeapi(
        pmeapi_defense, patch_adaptive_pmeapi, images, labels,
        label="PMEAPI-adaptive patch vs PMEAPI",
    )

    delta = res_std["recovered_acc"] - res_adp["recovered_acc"]
    n = res_std["n_images"]
    se = 100 * (0.5 / n ** 0.5)

    print("\n" + "=" * 70)
    print(f"  n={n}  SE~{se:.1f}pp")
    print(f"{'':32s} {'Oblivious':>15} {'PMEAPI-adaptive':>18}")
    print("-" * 70)
    print(f"  Undefended ASR:          {res_std['asr_undefended']:>14.1f}% {res_adp['asr_undefended']:>17.1f}%")
    print(f"  PMEAPI recovered acc:    {res_std['recovered_acc']:>14.1f}% {res_adp['recovered_acc']:>17.1f}%")
    print(f"  Mean posterior:          {res_std['mean_posterior']:>15.3f} {res_adp['mean_posterior']:>18.3f}")
    print(f"  Defense degradation:     {'—':>15} {(res_std['recovered_acc']-res_adp['recovered_acc']):>+17.1f}pp")
    print("=" * 70)

    sig = abs(delta) >= 2 * se
    print(f"\n{'[SIGNIFICANT]' if sig else '[NOT statistically significant]'} "
          f"at roughly n={n}, threshold~{2*se:.1f}pp (delta={delta:+.1f}pp).")

    return {"standard": res_std, "adaptive": res_adp, "delta_pp": delta, "significant": sig}


def _plot_losses(ce, kp, fs, gc, post):
    path = os.path.join(PLOTS_DIR, "adaptive_pmeapi_patch_losses.png")
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for ax, vals, title, color in zip(
        axes, [ce, kp, fs, gc, post],
        ["Classification loss (CE)", "TV loss (ORB proxy, not posterior)",
         "FS regulariser (raw L1)", "GradCAM concentration (GC)",
         "FS+GC partial posterior (direct)"],
        ["#D85A30", "#1D9E75", "#534AB7", "#EF6C00", "#6A1B9A"],
    ):
        ax.plot(vals, linewidth=0.8, color=color)
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel("Step")
        ax.grid(alpha=0.3)
    plt.suptitle("PMEAPI-targeted adaptive patch training losses", fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"PMEAPI-adaptive loss curves saved")