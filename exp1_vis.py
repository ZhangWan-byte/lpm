# exp1_vis.py
import os, json, glob, csv
import numpy as np
import matplotlib.pyplot as plt

# ---- import from your batch utilities (as requested) ----
from exp1_test_batch import procrustes_rmse  # aligns Z_hat to Z_true; returns (rmse, Z_aligned)

# =========================
# CONFIG — adjust if needed
# =========================
LP_ROOT = "./sim_data_batch/A1_poly_feats/"     # ground-truth latent positions
OUT_DIR = "./figs_A1_poly_feats_grids"
DPI = 300

# Fixed 3×4 layout and target Ds (with explicit run roots per panel)
TARGETS = [
    ("RG-VAE", 2, "./results/0905_1558_A1_poly_feats"),
    ("RG-VAE", 4, "./results/0904_2037_A1_poly_feats"),
    ("RG-VAE", 8, "./results/0903_2226_A1_poly_feats"),
    ("RG-VAE", 16, "./results/0905_0035_A1_poly_feats"),
    ("RG-VAE", 32, "./results/0903_2227_A1_poly_feats"),
    ("MLE", 2, "./results_mle/A1_poly_feats_D2"),
    ("MLE", 8, "./results_mle/A1_poly_feats_D8"),
    ("VI", 2, "./results_vi/A1_poly_feats_D2"),
    ("VI", 8, "./results_vi/A1_poly_feats_D8"),
    ("USVT", 2, "./results_usvt/A1_poly_feats_D2"),
    ("USVT", 8, "./results_usvt/A1_poly_feats_D8"),
    ("USVT", 16, "./results_usvt/A1_poly_feats_D16"),
]

# =========================
# Small, focused helpers
# =========================
INVALID_WIN_CHARS = r'<>:"/\\|?*'
def sanitize_filename(s: str) -> str:
    return "".join('_' if c in INVALID_WIN_CHARS else c for c in s)

def list_true_graphs_test_only(lp_root: str):
    """Return {graph: path} for TEST graphs only (names containing '_test_')."""
    patt = os.path.join(lp_root, "**", "*_nodes.npz")
    idx = {}
    for p in glob.iglob(patt, recursive=True):
        base = os.path.basename(p)  # e.g., A1_poly_feats_N100_test_00_nodes.npz
        graph = base.replace("_nodes.npz", "")
        if "_test_" in graph:
            idx[graph] = p
    return dict(sorted(idx.items()))

def _fmt(x, nd=4):
    try: return f"{float(x):.{nd}f}"
    except: return "N/A" if x is None else str(x)

def first_json_under(root_dir: str, names=("test_metrics.json",)):
    """Find a metrics JSON under root_dir. Prefer explicit names; else fall back to any test_metrics*.json."""
    for nm in names:
        p = os.path.join(root_dir, nm)
        if os.path.exists(p):
            return p
    candidates = sorted(glob.glob(os.path.join(root_dir, "test_metrics*.json")))
    if candidates:
        return candidates[0]
    candidates = sorted(glob.glob(os.path.join(root_dir, "**", "test_metrics*.json"), recursive=True))
    return candidates[0] if candidates else None

def _extract_auc_ap(detail: dict):
    """
    Pull per-graph AUC/AP with robust key names.
    Prefers *_1to1 variants; falls back to generic 'auc'/'ap' if needed.
    Also returns ap_all when present.
    """
    auc_1to1 = detail.get("auc_1to1")
    ap_1to1  = detail.get("ap_1to1")
    ap_all   = detail.get("ap_all")
    # Fallbacks
    if auc_1to1 is None:
        auc_1to1 = detail.get("auc")
    if ap_1to1 is None:
        ap_1to1 = detail.get("ap")
    return auc_1to1, ap_1to1, ap_all

# =========================
# Panel data accessors
# =========================
def get_rgvae_entry(graph: str, run_root: str):
    """
    Load per-graph metrics & zhat path from an RG-VAE run directory.
    Expects a test_metrics*.json with 'details' including this graph and a 'zhat_path' entry.
    """
    jp = first_json_under(run_root, names=("test_metrics.json",))
    if not jp:
        return None
    try:
        with open(jp, "r") as f:
            data = json.load(f)
    except Exception:
        return None
    for d in data.get("details", []):
        if d.get("graph") == graph:
            zrel = d.get("zhat_path")
            if not zrel:
                return None
            zhat_abs = zrel if os.path.isabs(zrel) else os.path.normpath(os.path.join(os.path.dirname(jp), zrel))
            auc_1to1, ap_1to1, ap_all = _extract_auc_ap(d)
            return {
                "gwd": d.get("gwd"),
                "lp_rmse": d.get("lp_rmse"),
                "auc_1to1": auc_1to1,
                "ap_1to1": ap_1to1,
                "ap_all": ap_all,
                "n_nodes": d.get("n_nodes"),
                "zhat_path": zhat_abs,
            }
    return None

def get_baseline_entry(graph: str, method: str, run_root: str):
    """
    Load per-graph metrics from the baseline's test_metrics JSON at run_root,
    and build the zhat file path per required patterns.
    """
    if method == "MLE":
        json_name = "test_metrics_mle.json"
        # MLE path: Zhat_mle_full/
        zhat_path = os.path.join(run_root, "Zhat_mle_full", f"{graph}_Zhat_mle.npy")
    elif method == "VI":
        json_name = "test_metrics_vi.json"
        zhat_path = os.path.join(run_root, "Zhat_vi_full", f"{graph}_Zhat_vi.npy")
    elif method == "USVT":
        json_name = "test_metrics_usvt.json"
        zhat_path = os.path.join(run_root, "Zhat_usvt_test", f"{graph}_Zhat_usvt.npy")
    else:
        return None

    jp = os.path.join(run_root, json_name)
    if not os.path.exists(jp):
        # fallback to any test_metrics*.json in the root (still prefers the expected name)
        jp = first_json_under(run_root, names=(json_name,))
        if not jp:
            return None

    try:
        with open(jp, "r") as f:
            data = json.load(f)
    except Exception:
        return None

    for d in data.get("details", []):
        if d.get("graph") == graph:
            auc_1to1, ap_1to1, ap_all = _extract_auc_ap(d)
            return {
                "gwd": d.get("gwd"),
                "lp_rmse": d.get("lp_rmse"),
                "auc_1to1": auc_1to1,
                "ap_1to1": ap_1to1,
                "ap_all": ap_all,
                "n_nodes": d.get("n_nodes"),
                "zhat_path": os.path.normpath(zhat_path),
            }
    return None

# =========================
# Rendering
# =========================
def render_grid_for_graph(graph: str, ztrue_nodes_npz: str, out_dir: str) -> str:
    Z_true = np.load(ztrue_nodes_npz)["positions"]

    fig, axes = plt.subplots(3, 4, figsize=(16, 12), dpi=DPI)
    axes = axes.flatten()

    for ax, (method, want_D, run_root) in zip(axes, TARGETS):
        if method == "RG-VAE":
            entry = get_rgvae_entry(graph, run_root)
        else:
            entry = get_baseline_entry(graph, method, run_root)

        if not entry:
            ax.set_title(f"{method}: D={want_D} (missing)", fontsize=9)
            ax.axis("off")
            continue

        zpath = entry["zhat_path"]
        if not os.path.exists(zpath):
            ax.set_title(f"{method}: D={want_D} (file missing)", fontsize=9)
            ax.axis("off")
            continue

        try:
            Z_hat = np.load(zpath)
            # Align using your utility
            _, Z_aligned = procrustes_rmse(Z_true, Z_hat, center=False, scale=False)

            # Plot first two dimensions for visualization
            Xt = Z_true if Z_true.shape[1] == 2 else Z_true[:, :2]
            Xh = Z_aligned if Z_aligned.shape[1] == 2 else Z_aligned[:, :2]

            ax.scatter(Xh[:, 0], Xh[:, 1], s=1, label="Z_hat")
            ax.scatter(Xt[:, 0], Xt[:, 1], s=1, label="Z_true")
            ax.legend(fontsize=8)
            ax.set_aspect("equal")

            gwd = _fmt(entry.get("gwd"))
            rmse_local = _fmt(entry.get("lp_rmse"))
            auc11 = _fmt(entry.get("auc_1to1"))
            ap11  = _fmt(entry.get("ap_1to1"))
            # Optional: show AP(all) in parentheses if present
            ap_all = entry.get("ap_all")
            ap_all_str = _fmt(ap_all) if ap_all is not None else None
            ap_line = f"AUC(1:1)={auc11} | AP(1:1)={ap11}"
            if ap_all_str is not None:
                ap_line += f" | AP(all)={ap_all_str}"

            n_nodes = entry.get("n_nodes", "?")
            ax.set_title(
                f"{method}: D={want_D}, n={n_nodes}\n"
                f"GWD={gwd} | LP-RMSE={rmse_local}\n"
                f"{ap_line}",
                fontsize=9
            )

        except Exception as e:
            ax.set_title(f"{method}: D={want_D} (error)", fontsize=9)
            ax.text(0.5, 0.5, str(e), ha="center", va="center", fontsize=8)
            ax.axis("off")

    fig.suptitle(f"{graph} — RG-VAE vs baselines (Ẑ vs Z_true) [TEST SET]", fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    os.makedirs(out_dir, exist_ok=True)
    fname = sanitize_filename(f"{graph}__grid.png")
    fpath = os.path.join(out_dir, fname)
    plt.savefig(fpath, bbox_inches="tight")
    plt.close(fig)
    return fpath

# =========================
# Main
# =========================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    grids_dir = os.path.join(OUT_DIR, "grids"); os.makedirs(grids_dir, exist_ok=True)

    # enumerate TEST graphs only
    graph_to_nodes = list_true_graphs_test_only(LP_ROOT)  # {graph: path_to_*_nodes.npz}

    rows = []
    for graph, nodes_npz in graph_to_nodes.items():
        out_path = render_grid_for_graph(graph, nodes_npz, grids_dir)
        rows.append([graph, os.path.relpath(out_path)])

    # manifest CSV
    csv_path = os.path.join(OUT_DIR, "_grid_index.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["graph", "grid_image_path"])
        w.writerows(rows)

    print(f"Saved {len(rows)} TEST grid figures to: {grids_dir}")
    print(f"Index CSV: {csv_path}")

if __name__ == "__main__":
    main()
