"""PT vs IT comparison figure for blame-recipient experiment."""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
PT_DIR = PROJECT_DIR / "results" / "blame_recipient_pt"
IT_DIR = PROJECT_DIR / "results" / "blame_recipient_it"
OUT    = PROJECT_DIR / "results" / "blame_recipient_it" / "pt_vs_it_compare.png"

REPORT_LAYERS = [8, 17, 25, 33]
COLORS_DEG = {1: "#a8d8ea", 2: "#5b9bd5", 3: "#2e75b6", 4: "#1f4e79", 5: "#0a1628"}
DEG_LABELS = {1:"DEG=1\nMild hint", 2:"DEG=2\nExplicit",
              3:"DEG=3\nCausal attr.", 4:"DEG=4\nStrong accuse", 5:"DEG=5\nGlobal condemn"}


def load(result_dir):
    acts = np.load(result_dir / "acts_all.npy").astype(np.float64)
    meta = json.loads((result_dir / "stimuli.json").read_text())
    blame_idx   = [i for i,m in enumerate(meta) if m["condition"]=="BLAME"]
    neutral_idx = [i for i,m in enumerate(meta) if m["condition"]=="NEUTRAL"]
    neutral_mean = np.nanmean(acts[neutral_idx], axis=0)
    deg_means = {d: np.nanmean(acts[[i for i in blame_idx if meta[i]["DEG"]==d]], axis=0)
                 for d in range(1, 6)}
    return acts, meta, deg_means, neutral_mean


def contrast_norm(deg_means, neutral_mean, layer):
    return {d: float(np.linalg.norm(deg_means[d][layer] - neutral_mean[layer]))
            for d in range(1, 6)}


def cosine_dist_from_deg1(deg_means, layer):
    ref = deg_means[1][layer]
    out = {}
    for d in range(1, 6):
        v = deg_means[d][layer]
        cos = np.dot(ref, v) / (np.linalg.norm(ref) * np.linalg.norm(v) + 1e-9)
        out[d] = float(1 - cos)
    return out


def norm_trajectory(deg_means, neutral_mean, n_layers):
    return {d: [float(np.linalg.norm(deg_means[d][l] - neutral_mean[l]))
                for l in range(n_layers)]
            for d in range(1, 6)}


def cosine_traj(deg_means, n_layers):
    """DEG=1 vs DEG=5 cosine distance per layer."""
    return [float(1 - np.dot(deg_means[1][l], deg_means[5][l]) /
                  (np.linalg.norm(deg_means[1][l]) * np.linalg.norm(deg_means[5][l]) + 1e-9))
            for l in range(n_layers)]


# ── Load ───────────────────────────────────────────────────────────────────────
_, _, pt_deg, pt_neu = load(PT_DIR)
_, _, it_deg, it_neu = load(IT_DIR)
n_layers = 34

# ── Figure ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 13))
gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.42, wspace=0.35)

fig.suptitle("PT vs IT — Blame-Recipient Experiment | Gemma 3 4B",
             fontsize=14, fontweight="bold", y=0.98)

RL_LABELS = {8:"L8\n(early)", 17:"L17\n(mid)", 25:"L25\n(late)", 33:"L33\n(final)"}
x_deg = list(range(1, 6))
x_tick_labels = [DEG_LABELS[d] for d in range(1, 6)]

# ── Row 0: Norm trajectory (all 34 layers) ─────────────────────────────────────
for col, (model_name, deg_means, neu) in enumerate([("PT", pt_deg, pt_neu),
                                                     ("IT", it_deg, it_neu)]):
    ax = fig.add_subplot(gs[0, col*2 : col*2+2])
    traj = norm_trajectory(deg_means, neu, n_layers)
    for d in range(1, 6):
        ax.plot(range(n_layers), traj[d], color=COLORS_DEG[d],
                linewidth=1.8, label=f"DEG={d}")
    for rl in REPORT_LAYERS:
        ax.axvline(rl, color="gray", linestyle=":", alpha=0.5, linewidth=1)
    ax.set_xlabel("Layer", fontsize=9)
    ax.set_ylabel("‖DEG_mean − neutral‖", fontsize=9)
    ax.set_title(f"{model_name} — Activation magnitude trajectory (all layers)", fontsize=10)
    ax.legend(fontsize=7, ncol=5, loc="upper left")
    ax.grid(alpha=0.25)

# ── Row 1: Bar charts at L25 and L33 ──────────────────────────────────────────
for col, (model_name, deg_means, neu) in enumerate([("PT", pt_deg, pt_neu),
                                                     ("IT", it_deg, it_neu)]):
    for sub, rl in enumerate([25, 33]):
        ax = fig.add_subplot(gs[1, col*2 + sub])
        norms = contrast_norm(deg_means, neu, rl)
        vals  = [norms[d] for d in range(1, 6)]
        colors = [COLORS_DEG[d] for d in range(1, 6)]
        bars = ax.bar(range(1, 6), vals, color=colors, alpha=0.85, width=0.6)
        ax.set_xticks(range(1, 6))
        ax.set_xticklabels([f"DEG={d}" for d in range(1, 6)], fontsize=7)
        ax.set_ylabel("‖contrast‖", fontsize=8)
        ax.set_title(f"{model_name} magnitude @ L{rl}", fontsize=10)
        ax.grid(alpha=0.25, axis="y")

# ── Row 2: Cosine dist from DEG=1 at each report layer ────────────────────────
ax_pt = fig.add_subplot(gs[2, 0:2])
ax_it = fig.add_subplot(gs[2, 2:4])

for ax, model_name, deg_means in [(ax_pt, "PT", pt_deg), (ax_it, "IT", it_deg)]:
    x = np.arange(len(REPORT_LAYERS))
    w = 0.15
    for k, d in enumerate(range(2, 6)):   # skip DEG=1 (always 0)
        vals = [cosine_dist_from_deg1(deg_means, rl)[d] for rl in REPORT_LAYERS]
        ax.bar(x + (k-1.5)*w, vals, width=w, color=COLORS_DEG[d],
               alpha=0.85, label=f"DEG={d}", linewidth=0)
    ax.set_xticks(x)
    ax.set_xticklabels([RL_LABELS[rl] for rl in REPORT_LAYERS], fontsize=8)
    ax.set_ylabel("Cosine dist from DEG=1", fontsize=9)
    ax.set_title(f"{model_name} — Directional divergence (vs DEG=1)", fontsize=10)
    ax.legend(fontsize=7, ncol=4)
    ax.grid(alpha=0.25, axis="y")
    ax.set_ylim(bottom=0)

plt.savefig(OUT, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {OUT}")
