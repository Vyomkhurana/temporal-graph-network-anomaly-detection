"""Streamlit dashboard for temporal graph anomaly detection outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(page_title="Temporal Graph Anomaly Dashboard", layout="wide")
st.title("Real-Time Network Lateral Movement Detection")
st.caption("Operational view for graph sanity, training metrics, and inference alerts")


def _load_json(path: str) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _load_csv(path: str) -> pd.DataFrame | None:
    p = Path(path)
    if not p.exists():
        return None
    return pd.read_csv(p, low_memory=False)


graph_summary = _load_json("data/processed/graph_sanity_summary.json")
metrics = _load_json("data/processed/metrics/baseline_metrics.json")
scored = _load_csv("data/processed/inference/scored_edges.csv")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Graph Health")
    if graph_summary:
        st.metric("Unique nodes", graph_summary.get("unique_nodes", "n/a"))
        st.metric("Time windows", graph_summary.get("time_windows", "n/a"))
        st.metric("Anomaly event rate", graph_summary.get("anomaly_event_rate", "n/a"))
    else:
        st.info("Run training pipeline first to generate graph summary.")

with col2:
    st.subheader("Validation")
    if metrics and "validation" in metrics:
        val = metrics["validation"]
        st.metric("AUC", f"{val.get('auc', 0.0):.4f}")
        st.metric("F1", f"{val.get('f1', 0.0):.4f}")
        st.metric("Threshold", f"{val.get('best_threshold', 0.5):.2f}")
    else:
        st.info("No validation metrics found yet.")

with col3:
    st.subheader("Test")
    if metrics and "test" in metrics:
        test = metrics["test"]
        st.metric("AUC", f"{test.get('auc', 0.0):.4f}")
        st.metric("Recall", f"{test.get('recall', 0.0):.4f}")
        st.metric("Precision", f"{test.get('precision', 0.0):.4f}")
    else:
        st.info("No test metrics found yet.")

st.divider()
st.subheader("Top Predicted Anomalies")

if scored is not None and not scored.empty:
    ranked = scored.sort_values("anomaly_probability", ascending=False).head(200)
    st.dataframe(ranked, use_container_width=True, height=420)

    chart_df = ranked[["anomaly_probability"]].reset_index(drop=True)
    st.line_chart(chart_df)
else:
    st.info("No scored inference output found. Run: python main.py --mode infer")

st.divider()
st.subheader("Runbook")
st.code(
    "\n".join(
        [
            "python main.py --mode train --epochs 5 --window 1h",
            "python main.py --mode infer",
            "streamlit run dashboard/app.py",
        ]
    ),
    language="bash",
)
