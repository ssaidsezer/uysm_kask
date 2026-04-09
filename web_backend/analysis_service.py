"""Sonuç Analizi — same computations as streamlit _render_analysis_tab; returns Plotly JSON."""

from __future__ import annotations

import json
from io import BytesIO
from typing import List

import pandas as pd
import plotly.graph_objects as go

from .schemas import AnalysisResponse


def _grouped_bar(
    pivot_df: pd.DataFrame, title: str, y_label: str, color_map: dict | None = None
) -> go.Figure:
    fig = go.Figure()
    # Dark-theme friendly, high-contrast palette for grouped bars.
    default_colors = ["#60A5FA", "#F59E0B", "#34D399", "#F472B6", "#A78BFA", "#F87171"]
    for i, col in enumerate(pivot_df.columns):
        color = (color_map or {}).get(col, default_colors[i % len(default_colors)])
        fig.add_trace(
            go.Bar(
                name=str(col),
                x=pivot_df.index.tolist(),
                y=pivot_df[col].tolist(),
                marker_color=color,
                text=[f"{v:.1f}" if pd.notna(v) else "" for v in pivot_df[col]],
                textposition="outside",
            )
        )
    fig.update_layout(
        barmode="group",
        title=title,
        yaxis_title=y_label,
        xaxis_title="Model",
        legend_title="RAG Tipi",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#fafafa"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
        margin=dict(t=60, b=40),
    )
    return fig


def _fig_to_json(fig: go.Figure) -> dict:
    return json.loads(fig.to_json())


def analyze_export_csv(content: bytes) -> AnalysisResponse:
    warnings: List[str] = []
    info_messages: List[str] = []
    charts: List[dict] = []

    try:
        df = pd.read_csv(BytesIO(content), sep=";")
    except Exception as e:
        return AnalysisResponse(warnings=[f"CSV okunamadı: {e}"])

    required_cols = {
        "model",
        "rag_type",
        "ai_score",
        "ai_verdict",
        "ai_hallucination_risk",
        "tokens_per_second",
    }
    missing = required_cols - set(df.columns)
    if missing:
        warnings.append(
            f"CSV'de eksik kolonlar: {', '.join(sorted(missing))}. Bazı metrikler hesaplanamayabilir."
        )

    for col in (
        "ai_score",
        "tokens_per_second",
        "response_time_seconds",
        "eval_duration_seconds",
        "retrieved_chunk_size",
    ):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    rag_vals = df["rag_type"].dropna().unique().tolist() if "rag_type" in df.columns else []
    rag_label = next((v for v in rag_vals if "siz" not in str(v).lower()), None)
    no_rag_label = next((v for v in rag_vals if "siz" in str(v).lower()), None)

    def _pct(part: float, total: float) -> float:
        return round(part / total * 100, 1) if total > 0 else 0.0

    def _group_metrics(grp: pd.DataFrame) -> dict:
        total = len(grp)
        out: dict = {"Satır": total}
        if "ai_score" in grp.columns:
            out["Ort. Skor"] = round(grp["ai_score"].mean(), 2)
        if "ai_verdict" in grp.columns:
            vc = grp["ai_verdict"].value_counts()
            out["Correct %"] = _pct(vc.get("correct", 0), total)
            out["Partial %"] = _pct(vc.get("partial", 0), total)
            out["Incorrect %"] = _pct(vc.get("incorrect", 0), total)
        if "ai_hallucination_risk" in grp.columns:
            hc = grp["ai_hallucination_risk"].value_counts()
            out["Halüsinasyon High %"] = _pct(hc.get("high", 0), total)
        if "tokens_per_second" in grp.columns:
            out["Ort. TPS"] = round(grp["tokens_per_second"].mean(), 1)
        if "response_time_seconds" in grp.columns:
            out["Ort. Yanıt (sn)"] = round(grp["response_time_seconds"].mean(), 2)
        return out

    global_summary: dict | None = None
    if rag_label and no_rag_label:
        rag_df = df[df["rag_type"] == rag_label]
        no_rag_df = df[df["rag_type"] == no_rag_label]
        gs: dict = {"row_count": len(df)}
        if "ai_score" in df.columns:
            rag_mean = rag_df["ai_score"].mean()
            no_rag_mean = no_rag_df["ai_score"].mean()
            delta = (
                (rag_mean - no_rag_mean) if pd.notna(rag_mean) and pd.notna(no_rag_mean) else None
            )
            gs["rag_mean_score"] = float(rag_mean) if pd.notna(rag_mean) else None
            gs["no_rag_mean_score"] = float(no_rag_mean) if pd.notna(no_rag_mean) else None
            gs["score_delta"] = float(delta) if delta is not None else None
        if "ai_verdict" in df.columns:
            gs["rag_correct_pct"] = _pct((rag_df["ai_verdict"] == "correct").sum(), len(rag_df))
            gs["no_rag_correct_pct"] = _pct((no_rag_df["ai_verdict"] == "correct").sum(), len(no_rag_df))
        if "ai_hallucination_risk" in df.columns:
            gs["rag_high_halluc_pct"] = _pct(
                (rag_df["ai_hallucination_risk"] == "high").sum(), len(rag_df)
            )
            gs["no_rag_high_halluc_pct"] = _pct(
                (no_rag_df["ai_hallucination_risk"] == "high").sum(), len(no_rag_df)
            )
        if "tokens_per_second" in df.columns:
            gs["rag_mean_tps"] = float(rag_df["tokens_per_second"].mean())
            gs["no_rag_mean_tps"] = float(no_rag_df["tokens_per_second"].mean())
        global_summary = gs
    else:
        info_messages.append("RAG'li / RAG'siz ayrımı tespit edilemedi — özet KPI'lar atlanıyor.")

    breakdown: List[dict] = []
    if "model" in df.columns and "rag_type" in df.columns:
        rows_breakdown = []
        for (model_name, rag_name), grp in df.groupby(["model", "rag_type"]):
            row = {"Model": model_name, "RAG": rag_name}
            row.update(_group_metrics(grp))
            rows_breakdown.append(row)
        breakdown_df = pd.DataFrame(rows_breakdown)
        breakdown_df = breakdown_df.sort_values(["Model", "RAG"]).reset_index(drop=True)
        breakdown = breakdown_df.to_dict(orient="records")

        if "Ort. Skor" in breakdown_df.columns:
            pivot_score = breakdown_df.pivot(index="Model", columns="RAG", values="Ort. Skor")
            charts.append(
                {
                    "id": "score_model_rag",
                    "title": "Ort. AI Skoru — Model × RAG",
                    "plotly": _fig_to_json(
                        _grouped_bar(pivot_score, "Ort. AI Skoru — Model × RAG", "Ort. Skor (0–10)")
                    ),
                }
            )
        if "Correct %" in breakdown_df.columns:
            pivot_correct = breakdown_df.pivot(index="Model", columns="RAG", values="Correct %")
            charts.append(
                {
                    "id": "correct_model_rag",
                    "title": "Correct % — Model × RAG",
                    "plotly": _fig_to_json(
                        _grouped_bar(pivot_correct, "Correct % — Model × RAG", "Correct %")
                    ),
                }
            )
        if "Ort. TPS" in breakdown_df.columns:
            pivot_tps = breakdown_df.pivot(index="Model", columns="RAG", values="Ort. TPS")
            charts.append(
                {
                    "id": "tps_model_rag",
                    "title": "Ort. Token/sn — Model × RAG",
                    "plotly": _fig_to_json(
                        _grouped_bar(pivot_tps, "Ort. Token/sn — Model × RAG", "Token / sn")
                    ),
                }
            )
    else:
        info_messages.append("'model' ve 'rag_type' kolonları gerekli.")

    lift_rows: List[dict] = []
    if rag_label and no_rag_label and "model" in df.columns:
        for model_name, grp in df.groupby("model"):
            rag_grp = grp[grp["rag_type"] == rag_label]
            no_rag_grp = grp[grp["rag_type"] == no_rag_label]
            if len(rag_grp) == 0 or len(no_rag_grp) == 0:
                continue
            lift_row: dict = {"Model": model_name}
            if "ai_score" in df.columns:
                rm = rag_grp["ai_score"].mean()
                nm = no_rag_grp["ai_score"].mean()
                lift_row["Skor Lift"] = round(rm - nm, 2) if pd.notna(rm) and pd.notna(nm) else None
            if "ai_verdict" in df.columns:
                rc = _pct((rag_grp["ai_verdict"] == "correct").sum(), len(rag_grp))
                nc = _pct((no_rag_grp["ai_verdict"] == "correct").sum(), len(no_rag_grp))
                lift_row["Correct % Lift"] = round(rc - nc, 1)
            if "ai_hallucination_risk" in df.columns:
                rh = _pct((rag_grp["ai_hallucination_risk"] == "high").sum(), len(rag_grp))
                nh = _pct((no_rag_grp["ai_hallucination_risk"] == "high").sum(), len(no_rag_grp))
                lift_row["Halüsinasyon Azalması (pp)"] = round(nh - rh, 1)
            if "tokens_per_second" in df.columns:
                rt = rag_grp["tokens_per_second"].mean()
                nt = no_rag_grp["tokens_per_second"].mean()
                lift_row["TPS Farkı"] = round(rt - nt, 1) if pd.notna(rt) and pd.notna(nt) else None
            if "question" in df.columns and "ai_score" in df.columns:
                merged = (
                    rag_grp[["question", "ai_score"]]
                    .rename(columns={"ai_score": "rag_score"})
                    .merge(
                        no_rag_grp[["question", "ai_score"]].rename(
                            columns={"ai_score": "no_rag_score"}
                        ),
                        on="question",
                        how="inner",
                    )
                )
                if len(merged) > 0:
                    lift_row["RAG'in Kötüleştirdiği %"] = _pct(
                        (merged["no_rag_score"] > merged["rag_score"]).sum(),
                        len(merged),
                    )
            lift_rows.append(lift_row)

        if lift_rows and "Skor Lift" in pd.DataFrame(lift_rows).columns:
            lift_df = pd.DataFrame(lift_rows)
            lift_pivot = lift_df.set_index("Model")[["Skor Lift"]]
            colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in lift_pivot["Skor Lift"]]
            fig_lift = go.Figure(
                go.Bar(
                    x=lift_pivot.index.tolist(),
                    y=lift_pivot["Skor Lift"].tolist(),
                    marker_color=colors,
                    text=[f"{v:+.2f}" if pd.notna(v) else "" for v in lift_pivot["Skor Lift"]],
                    textposition="outside",
                )
            )
            fig_lift.update_layout(
                title="Skor Lift — Model başına (RAG'li − RAG'siz)",
                yaxis_title="Lift",
                xaxis_title="Model",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#fafafa"),
                xaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
                yaxis=dict(
                    gridcolor="rgba(255,255,255,0.1)",
                    zeroline=True,
                    zerolinecolor="rgba(255,255,255,0.4)",
                ),
                showlegend=False,
                margin=dict(t=60, b=40),
            )
            charts.append({"id": "score_lift", "title": "Skor Lift", "plotly": _fig_to_json(fig_lift)})
        elif not lift_rows:
            info_messages.append(
                "Hiçbir model için hem RAG'li hem RAG'siz satır bulunamadı (RAG lift)."
            )
    elif rag_label is None or no_rag_label is None:
        pass
    else:
        info_messages.append("RAG lift için 'model' kolonu gerekli.")

    distribution_chart: dict | None = None
    if "ai_score" in df.columns and rag_label and no_rag_label:
        bins = list(range(0, 12))
        labels = [f"{i}" for i in range(0, 11)]
        dist_rows = []
        for rag_name in [rag_label, no_rag_label]:
            sub = df[df["rag_type"] == rag_name]["ai_score"].dropna()
            if len(sub) == 0:
                continue
            binned = pd.cut(sub, bins=bins, labels=labels, include_lowest=True, right=False)
            counts = binned.value_counts().reindex(labels, fill_value=0)
            for score_bin, cnt in counts.items():
                dist_rows.append({"Skor": score_bin, "RAG": rag_name, "Adet": int(cnt)})
        if dist_rows:
            dist_df = pd.DataFrame(dist_rows)
            pivot_dist = dist_df.pivot(index="Skor", columns="RAG", values="Adet").fillna(0)
            distribution_chart = _fig_to_json(
                _grouped_bar(pivot_dist, "Skor Dağılımı — RAG'li vs RAG'siz", "Soru Sayısı")
            )

    problems_rag_worse: List[dict] = []
    problems_both_low: List[dict] = []
    if (
        rag_label
        and no_rag_label
        and "ai_score" in df.columns
        and "question" in df.columns
        and "model" in df.columns
    ):
        rag_view = (
            df[df["rag_type"] == rag_label][["model", "question", "ai_score", "model_answer"]]
            .rename(columns={"ai_score": "rag_score", "model_answer": "rag_answer"})
        )
        no_rag_view = (
            df[df["rag_type"] == no_rag_label][["model", "question", "ai_score", "model_answer"]]
            .rename(columns={"ai_score": "no_rag_score", "model_answer": "no_rag_answer"})
        )
        merged_all = rag_view.merge(no_rag_view, on=["model", "question"], how="inner")
        problems_rag_worse = merged_all[
            merged_all["no_rag_score"] > merged_all["rag_score"]
        ].to_dict(orient="records")
        problems_both_low = merged_all[
            (merged_all["rag_score"] <= 5) & (merged_all["no_rag_score"] <= 5)
        ].to_dict(orient="records")

    display_cols = [
        c
        for c in [
            "model",
            "question",
            "rag_type",
            "ai_verdict",
            "ai_score",
            "ai_hallucination_risk",
            "tokens_per_second",
            "response_time_seconds",
            "eval_duration_seconds",
            "model_answer",
            "answer",
        ]
        if c in df.columns
    ]
    remaining = [c for c in df.columns if c not in display_cols]
    detail_df = df[display_cols + remaining].reset_index(drop=True)
    detail_rows = json.loads(detail_df.to_json(orient="records", default_handler=str))

    return AnalysisResponse(
        warnings=warnings,
        info_messages=info_messages,
        global_summary=global_summary,
        breakdown=breakdown,
        charts=charts,
        lift_rows=lift_rows,
        distribution_chart=distribution_chart,
        problems_rag_worse=problems_rag_worse,
        problems_both_low=problems_both_low,
        detail_rows=detail_rows,
        rag_label=str(rag_label) if rag_label is not None else None,
        no_rag_label=str(no_rag_label) if no_rag_label is not None else None,
    )
