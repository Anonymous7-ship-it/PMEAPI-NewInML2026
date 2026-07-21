import os
import numpy as np
from scipy.stats import beta as beta_dist
import matplotlib.pyplot as plt

# Use a local output directory (relative to current working directory)
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.grid": True,
    "grid.alpha": 0.3,
})
PARAMS = {
    "orb": {"alpha_0": 6.77, "beta_0": 20.57, "alpha_1": 20.15, "beta_1": 25.93,
            "mean_clean": 0.248, "mean_patched": 0.437, "title": "ORB keypoint density"},
    "fs":  {"alpha_0": 0.07, "beta_0": 2.20, "alpha_1": 0.71, "beta_1": 0.93,
            "mean_clean": 0.031, "mean_patched": 0.433, "title": "Feature squeezing"},
    "gc":  {"alpha_0": 7.25, "beta_0": 7.73, "alpha_1": 5.70, "beta_1": 1.53,
            "mean_clean": 0.484, "mean_patched": 0.788, "title": "GradCAM concentration"},
}
COLOR_CLEAN = "#D32F2F"
COLOR_PATCH = "#1565C0"


def plot_calibration():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    x = np.linspace(0.001, 0.999, 400)

    for ax, (sensor, p) in zip(axes, PARAMS.items()):
        pdf0 = beta_dist.pdf(x, p["alpha_0"], p["beta_0"])
        pdf1 = beta_dist.pdf(x, p["alpha_1"], p["beta_1"])
        ax.plot(x, pdf0, color=COLOR_CLEAN, linewidth=2,
                label=fr"Clean: Beta({p['alpha_0']:.2f}, {p['beta_0']:.2f})"
                      fr"  $\mu$={p['mean_clean']:.3f}")
        ax.fill_between(x, pdf0, color=COLOR_CLEAN, alpha=0.12)
        ax.plot(x, pdf1, color=COLOR_PATCH, linewidth=2,
                label=fr"Patched: Beta({p['alpha_1']:.2f}, {p['beta_1']:.2f})"
                      fr"  $\mu$={p['mean_patched']:.3f}")
        ax.fill_between(x, pdf1, color=COLOR_PATCH, alpha=0.12)
        ax.set_title(p["title"])
        ax.set_xlabel("Sensor score $s \\in [0,1]$")
        ax.set_ylabel("Density")
        ax.legend(fontsize=7.5, loc="upper center")
        ax.set_xlim(0, 1)

    fig.suptitle("PMEAPI Sensor Calibration: Beta Likelihood Fits "
                  "(clean vs. patched, 250-image calibration split)",
                  fontsize=11.5, y=1.04)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "pmeapi_calibration_final.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print("Calibration figure saved.")


# ── 2. Table II final comparison ─────────────────────────────────────────────
TABLE_II = {
    "no_defense":      {"label": "No defense",        "recovered": 15.7, "rec_lo": 13.6, "rec_hi": 17.9, "fpr": 0.0, "fpr_lo": 0.0, "fpr_hi": 0.0},
    "jpeg":            {"label": "JPEG compression",  "recovered": 53.9, "rec_lo": 50.6, "rec_hi": 57.0, "fpr": 0.2, "fpr_lo": 0.0, "fpr_hi": 0.5},
    "feature_squeeze": {"label": "Feature squeezing",  "recovered": 78.8, "rec_lo": 76.3, "rec_hi": 81.3, "fpr": 9.4, "fpr_lo": 7.7, "fpr_hi": 11.2},
    "gradcam_only":    {"label": "GradCAM only",       "recovered": 97.7, "rec_lo": 96.8, "rec_hi": 98.6, "fpr": 3.5, "fpr_lo": 2.4, "fpr_hi": 4.8},
    "patchcleanser":   {"label": "PatchCleanser (k=2)", "recovered": 65.1, "rec_lo": 62.0, "rec_hi": 68.1, "fpr": 0.4, "fpr_lo": 0.1, "fpr_hi": 0.8},
    "hybrid":          {"label": "Hybrid pipeline",     "recovered": 84.7, "rec_lo": 82.4, "rec_hi": 86.8, "fpr": 0.5, "fpr_lo": 0.1, "fpr_hi": 1.0},
    "pmeapi":          {"label": "PMEAPI (ours)",       "recovered": 98.2, "rec_lo": 97.3, "rec_hi": 99.0, "fpr": 0.6, "fpr_lo": 0.2, "fpr_hi": 1.1},
}
ORDER = ["no_defense", "jpeg", "feature_squeeze", "patchcleanser", "gradcam_only", "hybrid", "pmeapi"]


def plot_table2():
    methods = ORDER
    labels = [TABLE_II[m]["label"] for m in methods]
    rec = [TABLE_II[m]["recovered"] for m in methods]
    rec_err = np.array([[TABLE_II[m]["recovered"] - TABLE_II[m]["rec_lo"],
                          TABLE_II[m]["rec_hi"] - TABLE_II[m]["recovered"]] for m in methods]).T
    fpr = [TABLE_II[m]["fpr"] for m in methods]
    fpr_err = np.array([[TABLE_II[m]["fpr"] - TABLE_II[m]["fpr_lo"],
                          TABLE_II[m]["fpr_hi"] - TABLE_II[m]["fpr"]] for m in methods]).T

    colors = ["#9E9E9E" if m not in ("hybrid", "pmeapi")
              else ("#1565C0" if m == "hybrid" else "#EF6C00") for m in methods]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    y = np.arange(len(methods))
    axes[0].barh(y, rec, xerr=rec_err, color=colors, capsize=3,
                 error_kw={"linewidth": 1.2, "ecolor": "black"})
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=9.5)
    axes[0].set_xlabel("Recovered accuracy (%)")
    axes[0].set_title("Recovered Accuracy on Patched Images\n(1,000-image final_eval, 95% bootstrap CI)")
    axes[0].set_xlim(0, 105)
    for i, v in enumerate(rec):
        axes[0].text(v + 1.5, i, f"{v:.1f}%", va="center", fontsize=9)

    axes[1].barh(y, fpr, xerr=fpr_err, color=colors, capsize=3,
                 error_kw={"linewidth": 1.2, "ecolor": "black"})
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(labels, fontsize=9.5)
    axes[1].set_xlabel("False positive rate (%)")
    axes[1].set_title("False Positive Rate on Clean Images\n(1,000-image final_eval, 95% bootstrap CI)")
    axes[1].set_xlim(0, max(fpr) + 4)
    for i, v in enumerate(fpr):
        axes[1].text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=9)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "table2_final_comparison.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print("Table II comparison figure saved.")


if __name__ == "__main__":
    plot_calibration()
    plot_table2()