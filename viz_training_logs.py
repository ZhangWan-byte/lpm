#!/usr/bin/env python3
"""
Visualize RG-VAE training logs saved by exp1_train_batch.py with consistent run colors.

- Single run: saves inside that run's directory.
- Multiple runs: saves in a comparison directory (default results/<MMDD_HHMM>_compare or --out).
- Each run gets a base color + a lighter tint for train/val (and AUC/AP, GWD/LP-RMSE).

Usage:
  python viz_training_logs.py --results results/0828_2300
  python viz_training_logs.py --results results/0828_2300,results/0829_1015 --smooth
  python viz_training_logs.py --results results/0828_2300,results/0829_1015 --out results/my_compare --no_show
"""

import os
import argparse
from datetime import datetime
import json
import csv
import numpy as np
import matplotlib.pyplot as plt

# ---------- small utils ----------
def safe_get(d, key, default=None):
    return d[key] if key in d.files else default

def ema(x, alpha=0.2):
    x = np.asarray(x, dtype=float)
    if x.ndim == 0:
        return x
    y = np.empty_like(x, dtype=float)
    m = np.isnan(x)
    if np.all(m):
        return x
    last = np.nan
    for i in range(len(x)):
        if np.isnan(x[i]):
            y[i] = last if not np.isnan(last) else np.nan
        else:
            last = x[i]
            y[i] = x[i]
    out = np.copy(y)
    started = False
    for i, v in enumerate(y):
        if np.isnan(v):
            continue
        if not started:
            out[i] = v
            started = True
        else:
            out[i] = alpha * v + (1 - alpha) * out[i-1]
    return out

def load_metrics_npz(path_npz):
    data = np.load(path_npz, allow_pickle=True)
    return {
        "epochs":   safe_get(data, "epochs"),
        "tr_total": safe_get(data, "tr_total"),
        "tr_edge":  safe_get(data, "tr_edge"),
        "tr_feat":  safe_get(data, "tr_feat"),
        "tr_kl":    safe_get(data, "tr_kl"),
        "va_total": safe_get(data, "va_total"),
        "va_edge":  safe_get(data, "va_edge"),
        "va_feat":  safe_get(data, "va_feat"),
        "va_kl":    safe_get(data, "va_kl"),
        "va_auc":   safe_get(data, "va_auc"),
        "va_ap":    safe_get(data, "va_ap"),
        "gwd2":     safe_get(data, "gwd2"),
        "lp_rmse":  safe_get(data, "lp_rmse"),
        "setting_dir": safe_get(data, "setting_dir"),
    }

def summarize(metrics, name="run"):
    ep = metrics["epochs"]
    if ep is None: return None
    last = int(ep[-1])

    def last_valid(x):
        if x is None: return np.nan
        x = np.array(x, dtype=float)
        for v in x[::-1]:
            if not np.isnan(v): return float(v)
        return np.nan

    return {
        "run": name,
        "last_epoch": last,
        "train_total": last_valid(metrics["tr_total"]),
        "val_total": last_valid(metrics["va_total"]),
        "val_auc": last_valid(metrics["va_auc"]),
        "val_ap": last_valid(metrics["va_ap"]),
        "gwd2": last_valid(metrics["gwd2"]),
        "lp_rmse": last_valid(metrics["lp_rmse"]),
    }

# ---------- color handling ----------
def _hex_to_rgb01(hexstr):
    hexstr = hexstr.lstrip("#")
    return tuple(int(hexstr[i:i+2], 16)/255.0 for i in (0, 2, 4))

def _tint(rgb, amount=0.5):
    # blend towards white by 'amount' in [0,1]
    r, g, b = rgb
    return (r + (1 - r)*amount, g + (1 - g)*amount, b + (1 - b)*amount)

def build_run_palette(labels):
    """
    Assign each run label (directory basename) a base color and a light tint.
    We use a curated list for maximum separation, then cycle if needed.
    """
    # curated base palette (distinct, colorblind-aware leaning)
    base_hex = [
        "#D62728",  # red
        "#1F77B4",  # blue
        "#2CA02C",  # green
        "#9467BD",  # purple
        "#FF7F0E",  # orange
        "#17BECF",  # teal
        "#8C564B",  # brown
        "#E377C2",  # pink
        "#7F7F7F",  # gray
        "#BCBD22",  # olive
        "#AEC7E8",  # light blue (if many runs)
        "#FFBB78",  # light orange
    ]
    base_rgb = [_hex_to_rgb01(h) for h in base_hex]

    palette = {}
    n = len(labels)
    for i, lbl in enumerate(labels):
        base = base_rgb[i % len(base_rgb)]
        light = _tint(base, amount=0.55)  # lighter for val / secondary curves
        palette[lbl] = {"base": base, "light": light}
    return palette

# ---------- plotting with consistent colors ----------
def plot_losses(axs, epochs, tr_total, tr_edge, tr_feat, tr_kl, va_total, va_edge, va_feat, va_kl,
                run_label="", colors=None, smooth=False):
    c_base = colors[run_label]["base"] if colors else None
    c_light = colors[run_label]["light"] if colors else None

    if smooth:
        tr_total, tr_edge, tr_feat, tr_kl = map(ema, (tr_total, tr_edge, tr_feat, tr_kl))
        va_total, va_edge, va_feat, va_kl = map(ema, (va_total, va_edge, va_feat, va_kl))

    # Totals
    axs[0].plot(epochs, tr_total, color=c_base, label=f"{run_label} train")
    axs[0].plot(epochs, va_total,  color=c_light, linestyle="--", label=f"{run_label} val")
    axs[0].set_title("Total loss"); axs[0].set_xlabel("epoch"); axs[0].set_ylabel("loss"); axs[0].grid(True)

    # Edge
    axs[1].plot(epochs, tr_edge, color=c_base, label=f"{run_label} train")
    axs[1].plot(epochs, va_edge,  color=c_light, linestyle="--", label=f"{run_label} val")
    axs[1].set_title("Edge reconstruction (BCE)"); axs[1].set_xlabel("epoch"); axs[1].set_ylabel("loss"); axs[1].grid(True)

    # Feature
    axs[2].plot(epochs, tr_feat, color=c_base, label=f"{run_label} train")
    axs[2].plot(epochs, va_feat,  color=c_light, linestyle="--", label=f"{run_label} val")
    axs[2].set_title("Feature reconstruction"); axs[2].set_xlabel("epoch"); axs[2].set_ylabel("loss"); axs[2].grid(True)

    # KL
    axs[3].plot(epochs, tr_kl, color=c_base, label=f"{run_label} train")
    axs[3].plot(epochs, va_kl,  color=c_light, linestyle="--", label=f"{run_label} val")
    axs[3].set_title("KL divergence"); axs[3].set_xlabel("epoch"); axs[3].set_ylabel("loss"); axs[3].grid(True)

def plot_auc_ap(ax, epochs, auc, ap, run_label="", colors=None, smooth=False):
    c_base = colors[run_label]["base"] if colors else None
    c_light = colors[run_label]["light"] if colors else None
    if smooth:
        auc, ap = ema(auc), ema(ap)
    ax.plot(epochs, auc, color=c_base, label=f"{run_label} AUC")
    ax.plot(epochs, ap,  color=c_light, linestyle="--", label=f"{run_label} AP")
    ax.set_title("Link prediction (val)"); ax.set_xlabel("epoch"); ax.set_ylabel("score"); ax.grid(True)

def plot_geometry(ax, epochs, gwd2, lp_rmse, run_label="", colors=None, smooth=False):
    c_base = colors[run_label]["base"] if colors else None
    c_light = colors[run_label]["light"] if colors else None
    if gwd2 is not None:
        y = np.array(gwd2, dtype=float)
        mask = ~np.isnan(y)
        if mask.any():
            y_plot = ema(y) if smooth else y
            ax.plot(epochs[mask], y_plot[mask], color=c_base, label=f"{run_label} GWD$^2$")
    if lp_rmse is not None:
        y = np.array(lp_rmse, dtype=float)
        mask = ~np.isnan(y)
        if mask.any():
            y_plot = ema(y) if smooth else y
            ax.plot(epochs[mask], y_plot[mask], color=c_light, linestyle="--", label=f"{run_label} LP-RMSE")
    ax.set_title("Geometry metrics (val)"); ax.set_xlabel("epoch"); ax.set_ylabel("value"); ax.grid(True)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True,
                    help="Path to one results dir or comma-separated list (e.g., results/0828_2300[,results/0829_1015])")
    ap.add_argument("--smooth", action="store_true", help="Plot EMA-smoothed curves")
    ap.add_argument("--no_show", action="store_true", help="Do not open interactive windows, only save PNGs")
    ap.add_argument("--out", default=None, help="Output directory for comparison (only used when multiple runs)")
    args = ap.parse_args()

    runs = [r.strip() for r in args.results.split(",") if r.strip()]
    if not runs:
        raise SystemExit("No results directories provided.")

    # Load
    loaded = []
    for r in runs:
        npz_path = os.path.join(r, "metrics.npz")
        if not os.path.exists(npz_path):
            print(f"[WARN] Missing metrics.npz in {r} — skipping.")
            continue
        metrics = load_metrics_npz(npz_path)
        loaded.append((r, metrics))
    if not loaded:
        raise SystemExit("No valid runs found.")

    multi = len(loaded) > 1
    # Output dir
    if multi:
        out_dir = args.out or os.path.join("results", f"{datetime.now().strftime('%m%d_%H%M')}_compare")
        os.makedirs(out_dir, exist_ok=True)
        print(f"[INFO] Saving combined figures & summary to: {out_dir}")
    else:
        out_dir = loaded[0][0]

    # Build color palette: key by run label (directory basename)
    run_labels = [os.path.basename(os.path.normpath(rdir)) for (rdir, _) in loaded]
    colors = build_run_palette(run_labels)

    # ---- PLOTS: Losses
    fig1, axs1 = plt.subplots(2, 2, figsize=(12, 8))
    axs1 = axs1.ravel()
    for (rdir, m) in loaded:
        lbl = os.path.basename(os.path.normpath(rdir))
        plot_losses(axs1, m["epochs"], m["tr_total"], m["tr_edge"], m["tr_feat"], m["tr_kl"],
                    m["va_total"], m["va_edge"], m["va_feat"], m["va_kl"],
                    run_label=lbl, colors=colors, smooth=args.smooth)
    for ax in axs1: ax.legend(ncol=2, fontsize=9)
    fig1.suptitle("RG-VAE Training — Losses")
    fig1.tight_layout(rect=[0, 0.03, 1, 0.97])

    # ---- PLOTS: AUC/AP
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 5))
    for (rdir, m) in loaded:
        lbl = os.path.basename(os.path.normpath(rdir))
        plot_auc_ap(ax2, m["epochs"], m["va_auc"], m["va_ap"], run_label=lbl, colors=colors, smooth=args.smooth)
    ax2.legend(ncol=2, fontsize=9)
    fig2.suptitle("RG-VAE Training — Link Prediction (val)")
    fig2.tight_layout(rect=[0, 0.03, 1, 0.97])

    # ---- PLOTS: Geometry (GWD^2, LP-RMSE)
    fig3, ax3 = plt.subplots(1, 1, figsize=(10, 5))
    for (rdir, m) in loaded:
        lbl = os.path.basename(os.path.normpath(rdir))
        plot_geometry(ax3, m["epochs"], m["gwd2"], m["lp_rmse"], run_label=lbl, colors=colors, smooth=args.smooth)
    ax3.legend(ncol=2, fontsize=9)
    fig3.suptitle("RG-VAE Training — Geometry Metrics (val)")
    fig3.tight_layout(rect=[0, 0.03, 1, 0.97])

    # ---- Save figures
    losses_png   = os.path.join(out_dir, "losses.png")
    auc_ap_png   = os.path.join(out_dir, "auc_ap.png")
    geometry_png = os.path.join(out_dir, "geometry.png")
    fig1.savefig(losses_png, dpi=150)
    fig2.savefig(auc_ap_png, dpi=150)
    fig3.savefig(geometry_png, dpi=150)
    print(f"[SAVED] {losses_png}\n[SAVED] {auc_ap_png}\n[SAVED] {geometry_png}")

    # ---- Summaries
    summaries = []
    for (rdir, m) in loaded:
        base = os.path.basename(os.path.normpath(rdir))
        s = summarize(m, name=base)
        if s: summaries.append(s)

    summary_json = os.path.join(out_dir, "summary.json")
    with open(summary_json, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"[SAVED] {summary_json}")

    summary_csv = os.path.join(out_dir, "summary.csv")
    if summaries:
        keys = list(summaries[0].keys())
        with open(summary_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for row in summaries:
                w.writerow(row)
        print(f"[SAVED] {summary_csv}")

    if not args.no_show:
        plt.show()
    else:
        plt.close('all')

if __name__ == "__main__":
    main()
