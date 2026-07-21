import os
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import beta as beta_dist
from scipy.optimize import minimize_scalar
from scipy.special import expit as sigmoid
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.config import DEVICE, PLOTS_DIR, CHECKPOINT_DIR, IMAGE_SIZE


# Sensor score computation

def _orb_score(image: torch.Tensor) -> float:
    
    from src.classical import continuous_orb_score
    return continuous_orb_score(image)


def _fs_score_normalised(model, image: torch.Tensor) -> float:
    from src.feature_squeeze import compute_fs_score
    raw = compute_fs_score(model, image.unsqueeze(0)).item()
    return float(np.clip(raw / 1.75, 0.0, 1.0))  # 1.75 ~ empirical 99th pctile


def _gradcam_score(model, gradcam, image: torch.Tensor) -> float:
    try:
        with torch.no_grad():
            pred = model(image.unsqueeze(0).to(DEVICE)).argmax(1).item()

        heatmap = gradcam(image, pred)  # (224, 224) in [0, 1]

        if heatmap.max() == 0:
            return 0.0

        threshold = np.percentile(heatmap, 85)
        hot = heatmap >= threshold
        concentration = heatmap[hot].sum() / max(heatmap.sum(), 1e-8)
        return float(np.clip(concentration, 0.0, 1.0))
    except Exception:
        return 0.5  # uncertain on failure


def compute_sensor_scores(
    model, gradcam, image: torch.Tensor
) -> dict:
    """Compute all three continuous sensor scores for one image."""
    return {
        "orb": _orb_score(image),
        "fs":  _fs_score_normalised(model, image),
        "gc":  _gradcam_score(model, gradcam, image),
    }


# Beta distribution calibration 

def fit_beta(scores: np.ndarray) -> tuple[float, float]:
    """Fit a Beta(alpha, beta) distribution to a vector of scores in (0,1).

    Uses method-of-moments for robustness (MLE can diverge at 0 or 1).
    Clips scores to (eps, 1-eps) to keep them in the open unit interval.
    """
    eps = 1e-4
    scores = np.clip(scores, eps, 1 - eps)
    mu = scores.mean()
    var = scores.var()
    # Method-of-moments: mu = alpha/(alpha+beta), var = mu(1-mu)/(alpha+beta+1)
    if var <= 0 or var >= mu * (1 - mu):
        # Degenerate: return a near-uniform Beta
        return 1.0, 1.0
    concentration = mu * (1 - mu) / var - 1
    alpha = mu * concentration
    beta  = (1 - mu) * concentration
    return max(alpha, 0.01), max(beta, 0.01)


def calibrate(
    model,
    gradcam,
    images_clean:   torch.Tensor,
    images_patched: torch.Tensor,
    save: bool = True,
) -> dict:
    """
    Estimate Beta distribution parameters for each sensor on clean and
    patched images. This is the calibration step -- must be run on the
    fs_tuning partition, NOT on the final_eval partition.

    Returns a params dict:
      {
        "orb":  {"alpha_0": ..., "beta_0": ..., "alpha_1": ..., "beta_1": ...},
        "fs":   {...},
        "gc":   {...},
        "prior_log_odds": float,   # log P(Z=1)/P(Z=0) on deployment data
      }
    """
    N = images_clean.size(0)
    print(f"Calibrating PMEAPI on {N} clean + {N} patched images...")

    scores = {"clean": {"orb": [], "fs": [], "gc": []},
              "patched": {"orb": [], "fs": [], "gc": []}}

    for i in tqdm(range(N), desc="  Calibration (clean)"):
        s = compute_sensor_scores(model, gradcam, images_clean[i])
        for k in ["orb", "fs", "gc"]:
            scores["clean"][k].append(s[k])

    for i in tqdm(range(N), desc="  Calibration (patched)"):
        s = compute_sensor_scores(model, gradcam, images_patched[i])
        for k in ["orb", "fs", "gc"]:
            scores["patched"][k].append(s[k])

    params = {}
    for sensor in ["orb", "fs", "gc"]:
        a0, b0 = fit_beta(np.array(scores["clean"][sensor]))
        a1, b1 = fit_beta(np.array(scores["patched"][sensor]))
        params[sensor] = {
            "alpha_0": a0, "beta_0": b0,
            "alpha_1": a1, "beta_1": b1,
            "mean_clean":   np.mean(scores["clean"][sensor]),
            "mean_patched": np.mean(scores["patched"][sensor]),
        }
        print(f"  {sensor.upper():5s}: "
              f"clean Beta({a0:.2f},{b0:.2f}) mean={params[sensor]['mean_clean']:.3f} | "
              f"patched Beta({a1:.2f},{b1:.2f}) mean={params[sensor]['mean_patched']:.3f}")

    # Prior: assume equal class priors at deployment (1:1 clean:patched)
    # Users can override this to match their deployment base rate
    params["prior_log_odds"] = 0.0   # log(1/1) = 0

    params["_raw_scores"] = scores  # keep for diagnostic plots

    if save:
        path = os.path.join(CHECKPOINT_DIR, "pmeapi_params.pt")
        torch.save(params, path)
        print("Calibration params saved.")
        _plot_calibration(params, scores)

    return params


def load_calibration() -> dict:
    path = os.path.join(CHECKPOINT_DIR, "pmeapi_params.pt")
    if not os.path.exists(path):
        raise FileNotFoundError("Calibration file not found. Run calibrate() first.")
    return torch.load(path, map_location="cpu", weights_only=False)


# Posterior inference 

def _log_likelihood_ratio(score: float, alpha_0: float, beta_0: float,
                           alpha_1: float, beta_1: float) -> float:
    """Log likelihood ratio: log p(s|Z=1) - log p(s|Z=0).

    Uses the Beta PDF. Positive -> evidence for patch. Negative -> evidence
    for clean. Zero -> uninformative.
    """
    eps = 1e-10
    score = np.clip(score, eps, 1 - eps)
    log_p1 = beta_dist.logpdf(score, alpha_1, beta_1)
    log_p0 = beta_dist.logpdf(score, alpha_0, beta_0)
    return float(np.clip(log_p1 - log_p0, -20, 20))  # clip for numerical stability


def posterior(
    sensor_scores: dict,
    params: dict,
) -> float:
    """Compute P(Z=1 | s_orb, s_fs, s_gc) via log-odds fusion.

    log-odds(Z=1|s) = log-prior + sum_i log LR_i(s_i)
    P(Z=1|s) = sigma(log-odds(Z=1|s))

    This is exact under the assumption that sensor scores are conditionally
    independent given Z. We test this assumption during calibration and
    report the dependence statistics.
    """
    log_odds = params["prior_log_odds"]
    for sensor in ["orb", "fs", "gc"]:
        p = params[sensor]
        llr = _log_likelihood_ratio(
            sensor_scores[sensor],
            p["alpha_0"], p["beta_0"],
            p["alpha_1"], p["beta_1"],
        )
        log_odds += llr

    return float(sigmoid(log_odds))


# Soft masking 

def soft_mask_and_reclassify(
    model, gradcam,
    image: torch.Tensor,
    true_label: int,
    patch_prob: float,
    kp_bbox=None,
    soft_threshold: float = 0.5,
) -> dict:
    """
      - Low confidence (0.5-0.7): gentle masking, less clean-image damage
      - Moderate confidence (0.7-0.9): partial masking
      - High confidence (>0.9): full hard masking (original behaviour)
    """
    from src.gradcam_defense import (
        get_gradcam_bbox, mask_image, _intersect_bboxes
    )

    image = image.to(DEVICE)

    with torch.no_grad():
        pred_before = model(image.unsqueeze(0)).argmax(1).item()

    if patch_prob < soft_threshold:
        return {
            "predicted_after": pred_before,
            "correct_after": pred_before == true_label,
            "patch_prob": patch_prob,
            "masking_mode": "none",
        }

    try:
        heatmap, gradcam_bbox = get_gradcam_bbox(gradcam, image, pred_before)
    except Exception:
        return {
            "predicted_after": pred_before,
            "correct_after": pred_before == true_label,
            "patch_prob": patch_prob,
            "masking_mode": "gradcam_failed",
        }

    mask_bbox = _intersect_bboxes(gradcam_bbox, kp_bbox)
    if mask_bbox is None:
        return {
            "predicted_after": pred_before,
            "correct_after": pred_before == true_label,
            "patch_prob": patch_prob,
            "masking_mode": "no_bbox",
        }

    x_masked = mask_image(image, mask_bbox)

    # Soft blend: high patch_prob -> mostly masked; low -> mostly original
    if patch_prob >= 0.9:
        x_out = x_masked
        mode  = "hard"
    else:
        alpha = patch_prob
        x_out = alpha * x_masked + (1 - alpha) * image
        mode  = f"soft_{patch_prob:.2f}"

    with torch.no_grad():
        pred_after = model(x_out.unsqueeze(0)).argmax(1).item()

    return {
        "predicted_after": pred_after,
        "correct_after": pred_after == true_label,
        "patch_prob": patch_prob,
        "masking_mode": mode,
    }


# ── Full PMEAPI defense ───────────────────────────────────────────────────────

class PMEAPiDefense:
    """
    Probabilistic Multi-Evidence Adversarial Patch Inference (PMEAPI).

    Replaces the binary cascade with a principled posterior:
      1. Compute continuous sensor scores from ORB, FS, GradCAM.
      2. Fuse via log-likelihood ratios to get P(patch | evidence).
      3. Apply soft masking proportional to the posterior.
      4. Return calibrated confidence alongside the prediction.

    Compared to the binary HybridDefense:
      - Never makes an irreversible binary decision on ambiguous images.
      - Reports uncertainty so downstream systems can escalate to human
        review when the model is genuinely unsure.
      - FPR on clean images is lower because low-confidence detections
        get partial (not full) masking, reducing accuracy damage.
    """

    def __init__(self, model, gradcam, params: dict,
                 decision_threshold: float = 0.5):
        self.model = model
        self.gradcam = gradcam
        self.params = params
        self.decision_threshold = decision_threshold

    def defend(self, image: torch.Tensor, true_label: int = None) -> dict:
        image = image.to(DEVICE)

        # Stage 1: ORB keypoint score (cheap, runs first)
        orb_s = _orb_score(image)

        # Stage 2: Feature squeezing score
        fs_s = _fs_score_normalised(self.model, image)

        # Stage 3: GradCAM concentration score
        gc_s = _gradcam_score(self.model, self.gradcam, image)

        sensor_scores = {"orb": orb_s, "fs": fs_s, "gc": gc_s}
        patch_prob = posterior(sensor_scores, self.params)

        # Get ORB bbox for intersection (reuse existing detect_patch)
        from src.classical import detect_patch
        _, kp_bbox, _ = detect_patch(image, method="orb")

        result = soft_mask_and_reclassify(
            self.model, self.gradcam, image,
            true_label=true_label if true_label is not None else -1,
            patch_prob=patch_prob,
            kp_bbox=kp_bbox,
            soft_threshold=self.decision_threshold,
        )

        result["sensor_scores"] = sensor_scores
        result["patch_prob"] = patch_prob
        result["correct"] = result["correct_after"]
        result["prediction"] = result["predicted_after"]
        return result


def evaluate_pmeapi(
    model, gradcam, params,
    images_clean:   torch.Tensor,
    images_patched: torch.Tensor,
    labels:         torch.Tensor,
    decision_threshold: float = 0.5,
) -> dict:
    """Evaluate PMEAPI on clean and patched images.
    Returns a result dict compatible with run_full_evaluation's format
    so it can be added directly as a new row in Table II.
    """
    defense = PMEAPiDefense(model, gradcam, params, decision_threshold)
    N = labels.size(0)
    labels_cpu = labels.cpu()

    clean_correct   = 0
    patched_correct = 0
    patch_probs_clean   = []
    patch_probs_patched = []

    import torch as _torch
    for i in tqdm(range(N), desc="  PMEAPI (patched)"):
        res = defense.defend(images_patched[i], true_label=labels_cpu[i].item())
        if res["correct_after"]:
            patched_correct += 1
        patch_probs_patched.append(res["patch_prob"])
        if DEVICE.type == "mps":
            _torch.mps.empty_cache()

    for i in tqdm(range(N), desc="  PMEAPI (clean)"):
        res = defense.defend(images_clean[i], true_label=labels_cpu[i].item())
        if res["correct_after"]:
            clean_correct += 1
        patch_probs_clean.append(res["patch_prob"])
        if DEVICE.type == "mps":
            _torch.mps.empty_cache()

    fp = N - clean_correct

    from src.evaluate import bootstrap_ci_metric_pair
    ci = bootstrap_ci_metric_pair(
        clean_successes=clean_correct,
        patched_successes=patched_correct,
        fp_successes=fp,
        total=N,
    )

    result = {
        "clean_acc":     ci["clean_acc"]["value"],
        "clean_ci_low":  ci["clean_acc"]["ci_low"],
        "clean_ci_high": ci["clean_acc"]["ci_high"],
        "recovered_acc": ci["recovered_acc"]["value"],
        "rec_ci_low":    ci["recovered_acc"]["ci_low"],
        "rec_ci_high":   ci["recovered_acc"]["ci_high"],
        "attack_sr":     ci["attack_sr"]["value"],
        "attack_ci_low": ci["attack_sr"]["ci_low"],
        "attack_ci_high":ci["attack_sr"]["ci_high"],
        "fpr":           ci["fpr"]["value"],
        "fpr_ci_low":    ci["fpr"]["ci_low"],
        "fpr_ci_high":   ci["fpr"]["ci_high"],
        "_patch_probs_clean":   patch_probs_clean,
        "_patch_probs_patched": patch_probs_patched,
    }

    print(f"\nPMEAPI results (threshold={decision_threshold:.2f}):")
    print(f"  Clean accuracy:      {result['clean_acc']:.1f}%")
    print(f"  Recovered accuracy:  {result['recovered_acc']:.1f}%")
    print(f"  False positive rate: {result['fpr']:.1f}%")
    return result


def tune_decision_threshold(
    model, gradcam, params,
    images_clean: torch.Tensor,
    images_patched: torch.Tensor,
    target_fpr: float = 0.01,
) -> float:
    """Find the decision threshold that achieves target_fpr on the
    calibration set, maximising recovered accuracy subject to that
    constraint. Analogous to the binary pipeline's theta tuning but
    operating on the posterior P(Z=1|s) rather than a raw feature score.
    """
    N = images_clean.size(0)
    defense = PMEAPiDefense(model, gradcam, params, decision_threshold=0.0)

    probs_clean   = []
    probs_patched = []

    for i in tqdm(range(N), desc="  Computing posteriors (threshold sweep)"):
        r_c = defense.defend(images_clean[i])
        r_p = defense.defend(images_patched[i])
        probs_clean.append(r_c["patch_prob"])
        probs_patched.append(r_p["patch_prob"])

    probs_clean   = np.array(probs_clean)
    probs_patched = np.array(probs_patched)

    # BUG FIX: the sweep previously stopped at 0.9, but the fused posterior
    # sigmoid(sum of 3 LLRs, each clipped to +/-20) can legitimately range
    # all the way to ~1.0. If even a couple of clean images produce a
    # posterior above 0.9 (plausible given the FS sensor's degenerate
    # Beta(0.07, 2.20) clean fit), the true FPR<=target operating point
    # sits above the old sweep ceiling, so the loop silently found nothing
    # and fell back to a meaningless 0.5 default with 0% detection. The
    # symptom ("Optimal threshold: 0.500  Detection: 0.0%") was this range
    # bug, not a genuine calibration failure.
    #
    # Sweep the full open interval with finer resolution near 1.0, since
    # that's where the actual FPR-constrained operating point lives.
    candidate_thresholds = np.concatenate([
        np.linspace(0.01, 0.90, 90, endpoint=False),
        np.linspace(0.90, 0.999, 200),
    ])

    best_thresh = None
    best_det    = 0.0
    swept_fprs  = []

    for thresh in candidate_thresholds:
        fpr = (probs_clean >= thresh).mean()
        det = (probs_patched >= thresh).mean()
        swept_fprs.append(fpr)
        if fpr <= target_fpr and det > best_det:
            best_det    = det
            best_thresh = thresh

    if best_thresh is None:
        # Genuinely no operating point in [0,1] achieves target_fpr -- this
        # is now a real finding, not a range artifact, so surface it loudly
        # rather than silently defaulting.
        min_achievable_fpr = min(swept_fprs)
        raise RuntimeError(
            f"No threshold in [0.01, 0.999] achieves FPR <= {target_fpr:.3f} "
            f"on this {len(probs_clean)}-image threshold-tuning set. "
            f"Minimum achievable FPR at threshold=0.999 is {min_achievable_fpr:.3f}. "
            f"With N={len(probs_clean)}, the smallest nonzero FPR you can even "
            f"resolve is {1/len(probs_clean):.3f} -- consider raising target_fpr "
            f"or using more threshold-tuning images."
        )

    print(f"Optimal threshold: {best_thresh:.3f}  "
          f"Detection: {100*best_det:.1f}%  "
          f"FPR <= {100*target_fpr:.1f}%")

    # Save the REAL computed posterior arrays (not simulated/hardcoded) 
    save_path = os.path.join(CHECKPOINT_DIR, "pmeapi_threshold_posteriors.pt")
    torch.save({
        "probs_clean":   probs_clean,
        "probs_patched": probs_patched,
        "best_thresh":   float(best_thresh),
        "best_det":      float(best_det),
        "target_fpr":    float(target_fpr),
        "n_images":      int(N),
    }, save_path)
    print("Threshold-tuning posteriors saved.")

    return float(best_thresh)


# ── Diagnostics and plots ─────────────────────────────────────────────────────

def _plot_calibration(params: dict, scores: dict):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    sensors   = ["orb", "fs", "gc"]
    titles    = ["ORB keypoint density", "Feature squeezing", "GradCAM concentration"]
    colors    = {"clean": "#378ADD", "patched": "#D85A30"}

    for ax, sensor, title in zip(axes, sensors, titles):
        p = params[sensor]
        x = np.linspace(0.001, 0.999, 300)

        # Histogram of empirical scores
        for split in ["clean", "patched"]:
            ax.hist(scores[split][sensor], bins=30, alpha=0.35,
                    color=colors[split], density=True, label=f"{split} (empirical)")

        # Fitted Beta PDFs
        ax.plot(x, beta_dist.pdf(x, p["alpha_0"], p["beta_0"]),
                color=colors["clean"], linewidth=2,
                label=f"Beta({p['alpha_0']:.1f},{p['beta_0']:.1f})")
        ax.plot(x, beta_dist.pdf(x, p["alpha_1"], p["beta_1"]),
                color=colors["patched"], linewidth=2,
                label=f"Beta({p['alpha_1']:.1f},{p['beta_1']:.1f})")

        ax.set_title(title)
        ax.set_xlabel("Sensor score")
        ax.set_ylabel("Density")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    plt.suptitle("PMEAPI sensor calibration: Beta fits to clean/patched score distributions",
                 fontsize=11)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "pmeapi_calibration.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print("Calibration plot saved.")


def plot_posterior_distributions(
    patch_probs_clean: list,
    patch_probs_patched: list,
    threshold: float,
):
    """Plot the posterior P(Z=1|s) distributions on clean and patched
    images. This is Figure 6 equivalent for PMEAPI -- the probabilistic
    analogue of the feature squeezing score histogram.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(0, 1, 41)
    ax.hist(patch_probs_clean,   bins=bins, alpha=0.6,
            color="#378ADD", label="Clean images P(patch|evidence)")
    ax.hist(patch_probs_patched, bins=bins, alpha=0.6,
            color="#D85A30", label="Patched images P(patch|evidence)")
    ax.axvline(threshold, color="#534AB7", linewidth=2, linestyle="--",
               label=f"Decision threshold = {threshold:.2f}")
    ax.set_xlabel("Posterior P(patch | ORB, FS, GradCAM)")
    ax.set_ylabel("Count")
    ax.set_title("PMEAPI posterior distributions on clean and patched images")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "pmeapi_posteriors.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print("Posterior distribution plot saved.")


def plot_threshold_analysis(
    probs_clean: np.ndarray,
    probs_patched: np.ndarray,
    threshold: float,
    target_fpr: float = 0.01,
    save_path: str = None,
):
    
    from scipy.stats import gaussian_kde
    from scipy.special import logit

    probs_clean   = np.asarray(probs_clean)
    probs_patched = np.asarray(probs_patched)

    # Recover fused log-odds from posteriors for the left panel (log-odds
    # is a monotonic reparameterisation of the posterior, so this is exact,
    # not an approximation -- clip away from 0/1 to avoid +/-inf).
    eps = 1e-6
    log_odds_clean   = logit(np.clip(probs_clean, eps, 1 - eps))
    log_odds_patched = logit(np.clip(probs_patched, eps, 1 - eps))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    # --- Left: evidence distributions ---
    ax = axes[0]
    lo = min(log_odds_clean.min(), log_odds_patched.min())
    hi = max(log_odds_clean.max(), log_odds_patched.max())
    xs = np.linspace(lo, hi, 400)
    kde0 = gaussian_kde(log_odds_clean)
    kde1 = gaussian_kde(log_odds_patched)
    ax.plot(xs, 0.5 * kde0(xs), color="#D32F2F", linewidth=1.8,
            label=r"$P(H_0)\,p(\ell\,|\,H_0)$  [clean]")
    ax.fill_between(xs, 0.5 * kde0(xs), color="#D32F2F", alpha=0.15)
    ax.plot(xs, 0.5 * kde1(xs), color="#1565C0", linewidth=1.8,
            label=r"$P(H_1)\,p(\ell\,|\,H_1)$  [patched]")
    ax.fill_between(xs, 0.5 * kde1(xs), color="#1565C0", alpha=0.15)
    ax.set_xlabel(r"Fused log-odds  $\ell = \sum_i \mathrm{LLR}_i(s_i)$")
    ax.set_ylabel("Scaled probability density")
    ax.set_title("PMEAPI Fused Evidence Distributions (real posteriors)")
    ax.legend(fontsize=8.5, loc="upper left")
    ax.grid(alpha=0.3)

    # --- Right: probability vs threshold ---
    ax = axes[1]
    gammas = np.linspace(0.001, 0.999, 400)
    p_fa   = np.array([(probs_clean >= g).mean() for g in gammas])
    p_miss = np.array([(probs_patched < g).mean() for g in gammas])
    p_err  = 0.5 * p_fa + 0.5 * p_miss

    ax.plot(gammas, p_fa, color="#D32F2F", linewidth=1.8,
            label=r"$P_{\mathrm{false\ acceptance}}(\gamma)$")
    ax.plot(gammas, p_miss, color="#1565C0", linewidth=1.8,
            label=r"$P_{\mathrm{miss}}(\gamma)$")
    ax.plot(gammas, p_err, color="#2E7D32", linewidth=1.6, linestyle="--",
            label=r"$P_{\mathrm{error}}(\gamma)$")
    ax.axvline(threshold, color="#6A1B9A", linestyle=":", linewidth=2,
               label=fr"tuned $\gamma^*={threshold:.3f}$ (FPR$\leq${100*target_fpr:.0f}%)")
    ax.set_xlabel(r"Decision threshold $\gamma$")
    ax.set_ylabel("Probability")
    ax.set_title(r"PMEAPI: Probabilities vs Threshold $\gamma$ (real posteriors)")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=8, loc="center right")
    ax.grid(alpha=0.3)

    n = len(probs_clean)
    fig.suptitle(f"PMEAPI decision-threshold analysis (N={n} threshold-tuning images, "
                 f"actual computed posteriors)", fontsize=10.5, y=1.02, color="dimgrey")
    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(PLOTS_DIR, "pmeapi_threshold_analysis.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print("Threshold analysis plot saved.")
    return save_path


def test_conditional_independence(scores: dict) -> dict:
    """Test whether sensor scores are conditionally independent given Z.

    The posterior derivation assumes conditional independence. We test
    this by computing the partial correlation between sensor scores
    after conditioning on the class label (clean / patched). A small
    partial correlation supports the independence assumption.

    Returns a dict of pairwise correlations for each class.
    """
    from scipy.stats import pearsonr

    result = {}
    pairs = [("orb", "fs"), ("orb", "gc"), ("fs", "gc")]

    for split in ["clean", "patched"]:
        result[split] = {}
        s = scores[split]
        for a, b in pairs:
            r, p = pearsonr(s[a], s[b])
            result[split][f"{a}_{b}"] = {"r": round(r, 3), "p": round(p, 4)}
            print(f"  {split:8s} {a}-{b}: r={r:.3f}  p={p:.4f} "
                  f"{'(significant)' if p < 0.05 else '(not significant)'}")

    return result