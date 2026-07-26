"""Streamlit dashboard for temporal graph anomaly detection outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
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


def _fmt_float(value: object, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "n/a"


graph_summary = _load_json("data/processed/graph_sanity_summary.json")
metrics = _load_json("data/processed/metrics/baseline_metrics.json")
pipeline_report = _load_json("data/processed/reports/pipeline_run_report.json")
scored = _load_csv("data/processed/inference/scored_edges.csv")

with st.sidebar:
    st.header("Artifacts")
    auto_refresh = st.toggle("Auto refresh", value=False)
    st.write("Use this dashboard to inspect the latest offline pipeline outputs.")

if auto_refresh:
    st.rerun()

top_row = st.columns(4)

with top_row[0]:
    st.subheader("Graph Health")
    if graph_summary:
        st.metric("Unique nodes", graph_summary.get("unique_nodes", "n/a"))
        st.metric("Time windows", graph_summary.get("time_windows", "n/a"))
        st.metric("Anomaly event rate", _fmt_float(graph_summary.get("anomaly_event_rate")))
    else:
        st.info("Run the pipeline first to generate graph summary.")

with top_row[1]:
    st.subheader("Validation")
    if metrics and "validation" in metrics:
        val = metrics["validation"]
        st.metric("AUC", _fmt_float(val.get("auc")))
        st.metric("F1", _fmt_float(val.get("f1")))
        st.metric("Threshold", _fmt_float(val.get("best_threshold"), digits=2))
    else:
        st.info("No validation metrics found yet.")

with top_row[2]:
    st.subheader("Test")
    if metrics and "test" in metrics:
        test = metrics["test"]
        st.metric("AUC", _fmt_float(test.get("auc")))
        st.metric("Recall", _fmt_float(test.get("recall")))
        st.metric("Precision", _fmt_float(test.get("precision")))
    else:
        st.info("No test metrics found yet.")

with top_row[3]:
    st.subheader("Pipeline")
    if pipeline_report:
        st.metric("Has report", "Yes")
        st.metric("Scored file", "Yes" if scored is not None else "No")
        st.metric("Model ready", "Yes" if Path("models/edge_baseline.pt").exists() else "No")
    else:
        st.info("No pipeline run report found yet.")

st.divider()

left_col, right_col = st.columns([1.1, 0.9])

with left_col:
    st.subheader("Artifact Snapshot")
    if pipeline_report:
        report_frame = pd.DataFrame(
            [
                {"artifact": key, "value": value}
                for key, value in pipeline_report.items()
                if key != "metrics"
            ]
        )
        st.dataframe(report_frame, use_container_width=True, height=260)
    else:
        st.info("Run the pipeline to populate the artifact snapshot.")

    st.subheader("Metric Overview")
    if metrics and "validation" in metrics:
        metric_rows = []
        for split_name, split_metrics in metrics.items():
            for metric_name, metric_value in split_metrics.items():
                metric_rows.append(
                    {"split": split_name, "metric": metric_name, "value": metric_value}
                )
        metric_frame = pd.DataFrame(metric_rows)
        st.dataframe(metric_frame, use_container_width=True, height=240)

        chart_source = metric_frame[metric_frame["metric"].isin(["auc", "f1", "precision", "recall"])]
        if not chart_source.empty:
            fig = px.bar(
                chart_source,
                x="metric",
                y="value",
                color="split",
                barmode="group",
                title="Validation/Test Metric Comparison",
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No metrics found yet.")

with right_col:
    st.subheader("Top Predicted Anomalies")
    if scored is not None and not scored.empty:
        ranked = scored.sort_values("anomaly_probability", ascending=False)
        threshold = float(
            metrics.get("validation", {}).get("best_threshold", 0.5) if metrics else 0.5
        )
        min_prob = st.slider("Minimum anomaly probability", 0.0, 1.0, threshold, 0.01)
        display_rows = ranked[ranked["anomaly_probability"] >= min_prob].head(200)
        st.dataframe(display_rows, use_container_width=True, height=360)

        if not display_rows.empty:
            score_fig = px.histogram(
                display_rows,
                x="anomaly_probability",
                nbins=20,
                title="Anomaly Probability Distribution",
            )
            st.plotly_chart(score_fig, use_container_width=True)
    else:
        st.info("No scored inference output found. Run: python -m src.pipeline.run_full_pipeline")

st.divider()
st.subheader("Runbook")
st.code(
    "\n".join(
        [
            "python -m src.pipeline.run_full_pipeline --epochs 3 --batch-size 512 --device cuda",
            "python -m src.training.train_edge_baseline --epochs 5 --device cuda",
            "python -m src.inference.score_edges --input data/processed/splits/test_edges.csv",
            "streamlit run dashboard/app.py",
        ]
    ),
    language="bash",
)

if pipeline_report:
    st.divider()
    st.subheader("Pipeline Run Summary")
    st.json(pipeline_report)
