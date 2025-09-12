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

# RG-VAE search root (timestamped runs)
RGVAE_SEARCH_ROOT = "./results"

# Fixed 3×4 layout and target Ds
TARGETS = [
    ("RG-VAE", 2), ("RG-VAE", 4), ("RG-VAE", 8), ("RG-VAE", 16),
    ("RG-VAE", 32), ("MLE", 2), ("MLE", 8), ("VI", 2),
    ("VI", 8), ("USVT", 2), ("USVT", 8), ("USVT", 16),
]

# Baseline roots + explicit file patterns
BASELINES = {
    "MLE": {
        "root": "./results_mle",
        "json_name": "test_metrics_mle.json",
        "zhat_rel": "{root}/A1_poly_feats_D{D}/Zhat_mle_full/{graph}_Zhat_mle.npy",
        "json_path": "{root}/A1_poly_feats_D{D}/test_metrics_mle.json",
    },
    "VI": {
        "root": "./results_vi",
        "json_name": "test_metrics_vi.json",
        "zhat_rel": "{root}/A1_poly_feats_D{D}/Zhat_vi_full/{graph}_Zhat_vi.npy",
        "json_path": "{root}/A1_poly_feats_D{D}/test_metrics_vi.json",
    },
    "USVT": {
        "root": "./results_usvt",
        "json_name": "test_metrics_usvt.json",
        "zhat_rel": "{root}/A1_poly_feats_D{D}/Zhat_usvt_test/{graph}_Zhat_usvt.npy",
        "json_path": "{root}/A1_poly_feats_D{D}/test_metrics_usvt.json",
    },
}

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

def _fmt(x):
    try: return f"{float(x):.4f}"
    except: return str(x)

def resolve_rel(json_path: str, maybe_rel: str) -> str:
    p = os.path.normpath(maybe_rel)
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(os.path.dirname(json_path), p))

def zhat_dim(zhat_path: str):
    try:
        Z = np.load(zhat_path, mmap_mode="r")
        return int(Z.shape[1]) if Z.ndim == 2 else None
    except Exception:
        return None

# =========================
# Load metrics into indices
# =========================
def load_rgvae_store(root: str):
    """
    Returns: dict graph -> list of entries
    entry = {
        "gwd":..., "lp_rmse":..., "auc_1to1":..., "ap_1to1":...,
        "n_nodes":..., "zhat_path":..., "json_path":...
    }
    Only keeps entries whose graph name contains '_test_'.
    """
    store = {}
    jsons = glob.iglob(os.path.join(root, "**", "test_metrics*.json"), recursive=True)
    for jp in jsons:
        try:
            with open(jp, "r") as f: data = json.load(f)
        except Exception as e:
            print(f"[RGVAE WARN] skip {jp}: {e}"); continue
        for d in data.get("details", []):
            g = d.get("graph"); zrel = d.get("zhat_path")
            if not g or not zrel or "_test_" not in g:
                continue
            store.setdefault(g, []).append({
                "gwd": d.get("gwd"),
                "lp_rmse": d.get("lp_rmse"),
                "auc_1to1": d.get("auc_1to1"),
                "ap_1to1": d.get("ap_1to1"),
                "n_nodes": d.get("n_nodes"),
                "zhat_path": resolve_rel(jp, zrel),
                "json_path": jp,
            })
    return store

def load_baseline_store(method: str, cfg: dict):
    """
    Returns: dict D -> dict graph -> entry
    entry = {
        "gwd":..., "lp_rmse":..., "auc_1to1":..., "ap_1to1":...,
        "n_nodes":..., "zhat_path":..., "json_path":...
    }
    Only keeps entries whose graph name contains '_test_'.
    """
    out = {}
    for D in [2, 4, 8, 16, 32]:
        jp = cfg["json_path"].format(root=cfg["root"], D=D)
        if not os.path.exists(jp):
            continue
        try:
            with open(jp, "r") as f: data = json.load(f)
        except Exception as e:
            print(f"[{method} WARN] skip {jp}: {e}"); continue
        level = out.setdefault(D, {})
        for d in data.get("details", []):
            g = d.get("graph")
            if not g or "_test_" not in g:
                continue
            zpath = cfg["zhat_rel"].format(root=cfg["root"], D=D, graph=g)
            level[g] = {
                "gwd": d.get("gwd"),
                "lp_rmse": d.get("lp_rmse"),
                "auc_1to1": d.get("auc_1to1"),
                "ap_1to1": d.get("ap_1to1"),
                "n_nodes": d.get("n_nodes"),
                "zhat_path": os.path.normpath(zpath),
                "json_path": jp,
            }
    return out

# =========================
# Panel selection helpers
# =========================
def select_rgvae_entry(rgvae_store: dict, graph: str, want_D: int):
    """Pick RG-VAE entry for this graph whose Z_hat dimension == want_D."""
    for e in rgvae_store.get(graph, []):
        Df = zhat_dim(e["zhat_path"])
        if Df == want_D:
            return e
    return None

def select_baseline_entry(baseline_store: dict, graph: str, want_D: int):
    """For MLE/VI/USVT, entries are indexed by D and graph directly."""
    return baseline_store.get(want_D, {}).get(graph)

# =========================
# Rendering
# =========================
def render_grid_for_graph(graph: str,
                          ztrue_nodes_npz: str,
                          rgvae_store: dict,
                          mle_store: dict,
                          vi_store: dict,
                          usvt_store: dict,
                          out_dir: str) -> str:
    Z_true = np.load(ztrue_nodes_npz)["positions"]

    fig, axes = plt.subplots(3, 4, figsize=(16, 12), dpi=DPI)
    axes = axes.flatten()

    for ax, (method, want_D) in zip(axes, TARGETS):
        if method == "RG-VAE":
            entry = select_rgvae_entry(rgvae_store, graph, want_D)
        elif method == "MLE":
            entry = select_baseline_entry(mle_store, graph, want_D)
        elif method == "VI":
            entry = select_baseline_entry(vi_store, graph, want_D)
        elif method == "USVT":
            entry = select_baseline_entry(usvt_store, graph, want_D)
        else:
            entry = None

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
            ap11 = _fmt(entry.get("ap_1to1"))
            n_nodes = entry.get("n_nodes", "?")
            ax.set_title(
                f"{method}: D={want_D}, n={n_nodes}\n"
                f"GWD={gwd} | LP-RMSE={rmse_local}\n"
                f"AUC(1:1)={auc11} | AP(1:1)={ap11}",
                fontsize=9
            )

        except Exception as e:
            ax.set_title(f"{method}: D={want_D} (error)", fontsize=9)
            ax.text(0.5, 0.5, str(e), ha="center", va="center", fontsize=8)
            ax.axis("off")

    fig.suptitle(f"{graph} — RG-VAE vs baselines (Z_hat vs Z_true) [TEST SET]", fontsize=12)
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

    # 1) enumerate TEST graphs only from LP_ROOT
    graph_to_nodes = list_true_graphs_test_only(LP_ROOT)  # {graph: path_to_*_nodes.npz}

    # 2) load RG-VAE metrics (timestamped runs), TEST only
    rgvae_store = load_rgvae_store(RGVAE_SEARCH_ROOT)

    # 3) load baselines using explicit patterns, TEST only
    mle_store  = load_baseline_store("MLE",  BASELINES["MLE"])
    vi_store   = load_baseline_store("VI",   BASELINES["VI"])
    usvt_store = load_baseline_store("USVT", BASELINES["USVT"])

    # 4) render one 3×4 per TEST graph
    rows = []
    for graph, nodes_npz in graph_to_nodes.items():
        out_path = render_grid_for_graph(graph, nodes_npz,
                                         rgvae_store, mle_store, vi_store, usvt_store,
                                         grids_dir)
        rows.append([graph, os.path.relpath(out_path)])

    # 5) manifest CSV
    csv_path = os.path.join(OUT_DIR, "_grid_index.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["graph", "grid_image_path"])
        w.writerows(rows)

    print(f"Saved {len(rows)} TEST grid figures to: {grids_dir}")
    print(f"Index CSV: {csv_path}")

if __name__ == "__main__":
    main()
