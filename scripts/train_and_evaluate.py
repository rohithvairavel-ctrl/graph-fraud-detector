#!/usr/bin/env python3
"""End-to-end: load data → graph features → train → metrics → figures."""
from __future__ import annotations
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import DATA_DIR, FIGURES_DIR, MODELS_DIR, REPORTS_DIR
from src.graph_features import compute_graph_features
from src.label_propagation import run_label_propagation
from src.train import save_artifacts, train_models

def plot_roc_pr(pred: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    y = pred["is_fraud"].values
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, col, color in [("Tabular only", "score_tabular", "#6c757d"), ("Tabular + Graph", "score_graph", "#e63946")]:
        RocCurveDisplay.from_predictions(y, pred[col], name=name, ax=axes[0])
        axes[0].get_lines()[-1].set_color(color)
        PrecisionRecallDisplay.from_predictions(y, pred[col], name=name, ax=axes[1])
        axes[1].get_lines()[-1].set_color(color)
    axes[0].set_title("ROC Curve"); axes[1].set_title("Precision–Recall Curve")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "roc_pr_curves.png", dpi=140)
    fig.savefig(FIGURES_DIR / "roc_pr_curves.svg"); plt.close(fig)

def plot_feature_importance(fi: dict) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    items = sorted(fi["tabular_plus_graph"].items(), key=lambda x: x[1], reverse=True)[:15]
    labels, vals = zip(*items)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(range(len(labels)), vals[::-1], color="#457b9d")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels[::-1], fontsize=9)
    ax.set_xlabel("Importance"); ax.set_title("Top features — Tabular + Graph model")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_importance.png", dpi=140)
    fig.savefig(FIGURES_DIR / "feature_importance.svg"); plt.close(fig)

def plot_lift_bars(metrics: dict) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    names = ["ROC-AUC", "PR-AUC", "P@50"]
    tab = [metrics["tabular_only"]["roc_auc"], metrics["tabular_only"]["pr_auc"], metrics["tabular_only"]["precision_at_50"]]
    graph = [metrics["tabular_plus_graph"]["roc_auc"], metrics["tabular_plus_graph"]["pr_auc"], metrics["tabular_plus_graph"]["precision_at_50"]]
    x, w = np.arange(len(names)), 0.35
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - w / 2, tab, w, label="Tabular only", color="#6c757d")
    ax.bar(x + w / 2, graph, w, label="Tabular + Graph", color="#e63946")
    ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score"); ax.set_title("Graph features lift over tabular baseline"); ax.legend()
    for i, (a, b) in enumerate(zip(tab, graph)):
        ax.text(i + w / 2, b + 0.02, f"+{b-a:.3f}", ha="center", fontsize=8, color="#e63946")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "metrics_lift.png", dpi=140)
    fig.savefig(FIGURES_DIR / "metrics_lift.svg"); plt.close(fig)

def main() -> None:
    accounts = pd.read_csv(DATA_DIR / "accounts.csv")
    transfers = pd.read_csv(DATA_DIR / "transfers.csv")
    tabular = pd.read_csv(DATA_DIR / "account_features.csv")
    print("Computing graph features...")
    gfeat = compute_graph_features(transfers, tabular)
    df = tabular.merge(gfeat, on="account_id", how="left")
    if "is_fraud" not in df.columns:
        df = df.merge(accounts[["account_id", "is_fraud"]], on="account_id")
    df.to_csv(DATA_DIR / "full_features.csv", index=False)
    print("Training models...")
    result = train_models(df)
    save_artifacts(result)
    print("Running label propagation...")
    result["metrics"]["label_propagation"] = run_label_propagation(df)
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(result["metrics"], indent=2))
    plot_roc_pr(result["predictions"])
    plot_feature_importance(result["feature_importance"])
    plot_lift_bars(result["metrics"])
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "run_summary.json").write_text(json.dumps({"n_nodes": int(df.shape[0]), "n_edges": int(len(transfers)), "metrics": result["metrics"]}, indent=2))
    print(json.dumps(result["metrics"], indent=2))
    print(f"Artifacts in {MODELS_DIR} and {REPORTS_DIR}")

if __name__ == "__main__":
    main()
