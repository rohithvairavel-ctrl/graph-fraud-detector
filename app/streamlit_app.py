"""Streamlit explorer: flagged accounts, neighborhood graph, feature contributions."""
from __future__ import annotations
import json, sys
from pathlib import Path
import joblib, networkx as nx, numpy as np, pandas as pd, streamlit as st
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import DATA_DIR, MODELS_DIR, REPORTS_DIR, TABULAR_FEATURES, GRAPH_FEATURES
from src.graph_features import neighborhood_subgraph

st.set_page_config(page_title="Graph Fraud Detector", layout="wide", page_icon="🕸️")

@st.cache_data
def load_data():
    full = DATA_DIR / "full_features.csv"
    df = pd.read_csv(full if full.exists() else DATA_DIR / "account_features.csv")
    transfers = pd.read_csv(DATA_DIR / "transfers.csv")
    metrics = json.loads((REPORTS_DIR / "metrics.json").read_text()) if (REPORTS_DIR / "metrics.json").exists() else {}
    preds = pd.read_csv(REPORTS_DIR / "test_predictions.csv") if (REPORTS_DIR / "test_predictions.csv").exists() else None
    fi = json.loads((REPORTS_DIR / "feature_importance.json").read_text()) if (REPORTS_DIR / "feature_importance.json").exists() else {}
    model = joblib.load(MODELS_DIR / "model_tabular_graph.joblib") if (MODELS_DIR / "model_tabular_graph.joblib").exists() else None
    return df, transfers, metrics, preds, fi, model

def main():
    st.title("🕸️ Graph Fraud Detector")
    st.markdown("Catch **fraud rings** via graph features (degree, PageRank, Louvain, shared devices/IPs, cycles) vs tabular baseline.")
    df, transfers, metrics, preds, fi, model = load_data()
    if metrics:
        tab, graph, lift = metrics.get("tabular_only", {}), metrics.get("tabular_plus_graph", {}), metrics.get("lift", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Tabular ROC-AUC", f"{tab.get('roc_auc', 0):.3f}")
        c2.metric("Graph ROC-AUC", f"{graph.get('roc_auc', 0):.3f}", delta=f"+{lift.get('roc_auc_lift', 0):.3f}")
        c3.metric("Tabular PR-AUC", f"{tab.get('pr_auc', 0):.3f}")
        c4.metric("Graph PR-AUC", f"{graph.get('pr_auc', 0):.3f}", delta=f"+{lift.get('pr_auc_lift', 0):.3f}")
    view = st.sidebar.radio("View", ["Flagged accounts", "Account neighborhood", "Feature contributions", "Metrics"])
    feat_cols = [c for c in TABULAR_FEATURES + GRAPH_FEATURES if c in df.columns]
    df = df.copy()
    if model is not None and all(c in df.columns for c in feat_cols):
        df["fraud_score"] = model.predict_proba(df[feat_cols].fillna(0))[:, 1]
    elif preds is not None:
        df = df.merge(preds[["account_id", "score_graph"]].rename(columns={"score_graph": "fraud_score"}), on="account_id", how="left")
        df["fraud_score"] = df["fraud_score"].fillna(0)
    else:
        df["fraud_score"] = df.get("is_fraud", 0)

    if view == "Flagged accounts":
        st.subheader("Top flagged accounts")
        top_n = st.slider("Top N", 10, 100, 25)
        cols = [c for c in ["account_id", "fraud_score", "is_fraud", "fraud_ring_id", "degree", "pagerank", "shared_device_count", "shared_ip_count", "cycle_flag", "community_size", "night_tx_ratio"] if c in df.columns]
        top = df.nlargest(top_n, "fraud_score")[cols]
        st.dataframe(top, use_container_width=True)
        if "is_fraud" in top.columns:
            st.caption(f"Precision@{top_n}: {top['is_fraud'].mean():.1%} true fraud")
    elif view == "Account neighborhood":
        st.subheader("Graph neighborhood")
        candidates = df.nlargest(50, "fraud_score")["account_id"].tolist()
        account_id = st.selectbox("Account", candidates)
        hops = st.slider("Hops", 1, 2, 1)
        sub = neighborhood_subgraph(transfers, account_id, hops=hops)
        st.write(f"**{sub.number_of_nodes()}** nodes, **{sub.number_of_edges()}** edges")
        fig, ax = plt.subplots(figsize=(8, 6))
        pos = nx.spring_layout(sub, seed=42)
        colors = []
        for n in sub.nodes():
            row = df.loc[df["account_id"] == n]
            if n == account_id: colors.append("#e63946")
            elif len(row) and row.iloc[0].get("is_fraud", 0) == 1: colors.append("#f4a261")
            else: colors.append("#457b9d")
        nx.draw_networkx(sub, pos, ax=ax, node_color=colors, node_size=280, font_size=7, arrows=True, edge_color="#adb5bd")
        ax.set_axis_off(); ax.set_title(f"{hops}-hop neighborhood of {account_id}")
        st.pyplot(fig); plt.close(fig)
    elif view == "Feature contributions":
        st.subheader("Feature contributions")
        if not fi:
            st.warning("Train models first.")
        else:
            which = st.selectbox("Model", list(fi.keys()))
            items = sorted(fi[which].items(), key=lambda x: x[1], reverse=True)[:20]
            labels, vals = zip(*items)
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.barh(range(len(labels)), list(vals)[::-1], color="#457b9d")
            ax.set_yticks(range(len(labels))); ax.set_yticklabels(list(labels)[::-1]); ax.set_xlabel("Importance")
            st.pyplot(fig); plt.close(fig)
            account_id = st.selectbox("Account profile", df.nlargest(30, "fraud_score")["account_id"].tolist())
            row = df.loc[df["account_id"] == account_id].iloc[0]
            imp = fi.get("tabular_plus_graph", fi[which])
            contrib = {c: abs(float(row[c])) * imp.get(c, 0.0) for c in feat_cols if c in row.index and c in imp}
            st.dataframe(pd.DataFrame(sorted(contrib.items(), key=lambda x: x[1], reverse=True)[:10], columns=["feature", "contribution"]), use_container_width=True)
    else:
        st.subheader("Held-out metrics"); st.json(metrics)
        for img in ["metrics_lift.png", "roc_pr_curves.png"]:
            p = REPORTS_DIR / "figures" / img
            if p.exists(): st.image(str(p))

if __name__ == "__main__":
    main()
