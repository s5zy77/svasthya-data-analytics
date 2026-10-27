"""
dashboard.py
============
Svasthya analytics dashboard (Streamlit + Plotly, light theme).

Run:  streamlit run dashboard.py

Pages
-----
1. Executive Overview & Health Trends
2. Symptom Co-occurrence & Data Mining
3. NLP Patient Triage Simulator

All heavy computations are wrapped in ``st.cache_data`` / ``st.cache_resource`` so the app stays
responsive when widgets change.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import nlp_analytics as nlp
from data_generator import CITY_META, generate_dataset

DATA_PATH = Path(__file__).parent / "data" / "svasthya_posts.csv"

# ---- Light-theme palette (one place to restyle) ------------------------------------------
PRIMARY = "#0F766E"      # teal
ACCENT = "#E11D48"       # rose, used for alerts
SEV_COLORS = {"Mild": "#16A34A", "Moderate": "#CA8A04", "Severe": "#EA580C", "Emergency": "#DC2626"}
PX_TEMPLATE = "plotly_white"
SEQ_SCALE = "Teal"

st.set_page_config(page_title="Svasthya Analytics", page_icon="🩺", layout="wide")

st.markdown(
    f"""
    <style>
      .block-container {{ padding-top: 1.6rem; }}
      div[data-testid="stMetric"] {{
          background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 12px; padding: 14px 16px;
      }}
      .card {{ background:#FFFFFF; border:1px solid #E2E8F0; border-radius:14px; padding:18px 20px;
               box-shadow:0 1px 3px rgba(15,23,42,.06); margin-bottom:12px; }}
      .badge {{ display:inline-block; padding:4px 14px; border-radius:999px; color:#fff;
                font-weight:700; font-size:1.05rem; }}
      .chip {{ display:inline-block; background:#F0FDFA; color:{PRIMARY}; border:1px solid #99F6E4;
               border-radius:999px; padding:2px 12px; margin:2px 4px 2px 0; font-size:.88rem; }}
      .tag {{ display:inline-block; background:#EEF2FF; color:#4338CA; border-radius:6px;
              padding:2px 10px; margin:2px 4px 2px 0; font-size:.88rem; font-family:monospace; }}
      .crisis {{ background:#FEF2F2; border:1px solid #FCA5A5; color:#991B1B; border-radius:12px;
                 padding:12px 16px; margin-bottom:12px; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================================================
# Cached data / model layer
# =========================================================================================
@st.cache_data(show_spinner="Loading and enriching posts…")
def load_data() -> pd.DataFrame:
    """Load the CSV (generate it on first run) and add all NLP features."""
    if DATA_PATH.exists():
        raw = pd.read_csv(DATA_PATH)
    else:
        raw = generate_dataset(1200, 42)
        DATA_PATH.parent.mkdir(exist_ok=True)
        raw.to_csv(DATA_PATH, index=False)
    return nlp.enrich_posts(raw)


@st.cache_data(show_spinner="Running outbreak detection…")
def get_signals(df: pd.DataFrame, z: float) -> pd.DataFrame:
    """Outbreak signals on the FULL dataset (baselines need history, so never on filtered data)."""
    return nlp.detect_outbreaks(df, z_threshold=z)


@st.cache_data(show_spinner="Clustering…")
def get_clusters(df: pd.DataFrame, k: int) -> nlp.ClusterResult:
    return nlp.cluster_posts(df, k=k)


@st.cache_data(show_spinner="Projecting to 2-D…")
def get_projection(X: np.ndarray, method: str) -> np.ndarray:
    return nlp.project_2d(X, method)


@st.cache_data(show_spinner=False)
def get_k_sweep(df: pd.DataFrame) -> pd.DataFrame:
    X, _, _ = nlp.build_embedding(df["clean_text"].tolist())
    return nlp.evaluate_k(X)


@st.cache_resource(show_spinner="Training triage model…")
def get_triage_model(df: pd.DataFrame) -> nlp.TriageClassifier:
    return nlp.TriageClassifier().fit(df)


df = load_data()

# =========================================================================================
# Sidebar
# =========================================================================================
st.sidebar.title("🩺 Svasthya")
st.sidebar.caption("Anonymous patient community & triage analytics")
page = st.sidebar.radio("Page", ["Overview & Trends", "Symptom Mining", "Triage Simulator"], label_visibility="collapsed")
st.sidebar.divider()
st.sidebar.caption("Synthetic data only. Educational project, not medical advice.")


# =========================================================================================
# PAGE 1: Executive overview
# =========================================================================================
def page_overview() -> None:
    st.title("Executive Overview & Health Trends")

    with st.sidebar:
        st.subheader("Filters")
        dmin, dmax = df["timestamp"].min().date(), df["timestamp"].max().date()
        date_range = st.date_input("Date range", (dmin, dmax), min_value=dmin, max_value=dmax)
        cities = st.multiselect("Cities", sorted(df["city"].unique()), default=sorted(df["city"].unique()))
        z_thr = st.slider("Outbreak z-score threshold", 1.5, 5.0, 2.5, 0.1,
                          help="Higher = fewer, more certain alerts. Robust z vs. 8-week rolling median baseline.")

    lo, hi = (date_range if isinstance(date_range, tuple) and len(date_range) == 2 else (dmin, dmax))
    view = df[(df["timestamp"].dt.date >= lo) & (df["timestamp"].dt.date <= hi) & df["city"].isin(cities)]
    if view.empty:
        st.warning("No posts match the current filters.")
        return

    signals = get_signals(df, z_thr)
    alerts = nlp.active_alerts(signals)
    alerts = alerts[alerts["city"].isin(cities)]

    patient_posts = view[view["user_type"] != "DOCTOR_VERIFIED"]
    engagement = patient_posts["doctor_replied"].mean() if len(patient_posts) else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total patient posts", f"{len(view):,}")
    c2.metric("Outbreak risk alerts", f"{len(alerts)}", help="Active (region, hashtag) alerts in the last 3 days of data.")
    c3.metric("Doctor engagement rate", f"{engagement:.0%}", help="Share of non-doctor posts that received a verified-doctor reply.")
    c4.metric("Avg triage severity", f"{view['severity_num'].mean():.2f} / 4", help="1 = Mild … 4 = Emergency")

    if not alerts.empty:
        top = alerts.iloc[0]
        st.error(f"🚨 Highest-risk signal: **{top['tag']}** in **{top['city']}**: {int(top['count_7d'])} posts in 7 days "
                 f"vs. a baseline of ~{top['baseline']:.0f} (z = {top['peak_z']:.1f}).")

    # ---- Time series ----------------------------------------------------------------------
    st.subheader("Daily post frequency by hashtag")
    exploded = view[["timestamp", "hashtag_list"]].explode("hashtag_list").dropna()
    exploded["date"] = exploded["timestamp"].dt.normalize()
    top_tags = exploded["hashtag_list"].value_counts().index.tolist()
    default_tags = [t for t in ["#fever", "#headache", "#dengue"] if t in top_tags]
    col_a, col_b = st.columns([3, 1])
    chosen = col_a.multiselect("Hashtags", top_tags[:15], default=default_tags)
    smooth = col_b.checkbox("7-day moving average", value=True)
    if chosen:
        ts = (exploded[exploded["hashtag_list"].isin(chosen)].groupby(["date", "hashtag_list"]).size()
              .unstack(fill_value=0).reindex(pd.date_range(lo, hi, freq="D"), fill_value=0))
        if smooth:
            ts = ts.rolling(7, min_periods=1).mean()
        long = ts.reset_index(names="date").melt("date", var_name="hashtag", value_name="posts")
        fig = px.line(long, x="date", y="posts", color="hashtag", template=PX_TEMPLATE,
                      labels={"posts": "posts / day" + (" (7d avg)" if smooth else "")})
        fig.update_layout(height=380, legend_title_text="", margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified")
        st.plotly_chart(fig, width="stretch")

    # ---- Regional heatmap + map -----------------------------------------------------------
    st.subheader("Regional outbreak intensity")
    st.caption("Intensity = latest robust z-score of the 7-day count vs. each region's own baseline (0 = normal). "
               "Computed on the full dataset.")
    inten = nlp.latest_intensity(signals)
    inten = inten.loc[[c for c in inten.index if c in cities]]
    left, right = st.columns([3, 2])
    with left:
        hm = px.imshow(inten, aspect="auto", color_continuous_scale="YlOrRd", template=PX_TEMPLATE,
                       labels=dict(x="Hashtag", y="City", color="z-score"))
        hm.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(hm, width="stretch")
    with right:
        city_score = inten.max(axis=1).rename("risk").reset_index().rename(columns={inten.index.name or "index": "city"})
        city_score.columns = ["city", "risk"]
        city_score["lat"] = city_score["city"].map(lambda c: CITY_META[c]["lat"])
        city_score["lon"] = city_score["city"].map(lambda c: CITY_META[c]["lon"])
        city_score["size"] = city_score["risk"].clip(lower=0) + 3
        mp = px.scatter_geo(city_score, lat="lat", lon="lon", size="size", color="risk", hover_name="city",
                            color_continuous_scale="YlOrRd", template=PX_TEMPLATE, scope="asia",
                            hover_data={"lat": False, "lon": False, "size": False, "risk": ":.1f"})
        mp.update_geos(lonaxis_range=[67, 95], lataxis_range=[6, 34], showcountries=True, countrycolor="#CBD5E1",
                       showland=True, landcolor="#F8FAFC")
        mp.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0), coloraxis_showscale=False)
        st.plotly_chart(mp, width="stretch")

    st.subheader("Active outbreak alerts")
    if alerts.empty:
        st.success("No active alerts at this threshold.")
    else:
        st.dataframe(alerts.rename(columns={"tag": "hashtag", "count_7d": "posts (7d)", "peak_z": "peak z"})
                     .round({"baseline": 1, "peak z": 1}), width="stretch", hide_index=True)


# =========================================================================================
# PAGE 2: Symptom mining
# =========================================================================================
def circular_layout(n: int) -> np.ndarray:
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.c_[np.cos(ang), np.sin(ang)]


def page_mining() -> None:
    st.title("Symptom Co-occurrence & Data Mining")
    tab1, tab2 = st.tabs(["Co-occurrence analysis", "Cluster explorer"])

    # ---- Co-occurrence --------------------------------------------------------------------
    with tab1:
        binary = nlp.hashtag_matrix(df, min_count=15)
        metrics = nlp.cooccurrence_metrics(binary)
        c1, c2 = st.columns([1, 3])
        metric = c1.selectbox("Metric", ["phi", "jaccard", "lift", "counts"],
                              help="phi = correlation of binary tags; jaccard = overlap; lift = association vs. chance.")
        c1.markdown({"phi": "**phi**: Pearson correlation for binary variables, corrects for popularity, can be negative.",
                     "jaccard": "**Jaccard**: |A∩B| / |A∪B|, intuitive overlap in [0, 1].",
                     "lift": "**Lift**: P(A,B) / (P(A)·P(B)). Above 1 means more than chance.",
                     "counts": "**Counts**: raw number of posts containing both tags."}[metric])
        mat = metrics[metric]
        fig = px.imshow(mat, color_continuous_scale="RdBu_r" if metric == "phi" else SEQ_SCALE,
                        zmin=-1 if metric == "phi" else None, zmax=1 if metric == "phi" else None,
                        template=PX_TEMPLATE, aspect="auto")
        fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10))
        c2.plotly_chart(fig, width="stretch")

        st.subheader("Symptom network")
        st.caption("Nodes = hashtags (size ∝ posts). Edges = Jaccard overlap above the threshold.")
        thr = st.slider("Minimum Jaccard overlap to draw an edge", 0.02, 0.6, 0.10, 0.01)
        tags = list(binary.columns)
        pos = circular_layout(len(tags))
        freq = binary.sum().to_numpy()
        jac = metrics["jaccard"].fillna(0).to_numpy()
        fig = go.Figure()
        for i in range(len(tags)):
            for j in range(i + 1, len(tags)):
                if jac[i, j] >= thr:
                    fig.add_trace(go.Scatter(x=[pos[i, 0], pos[j, 0]], y=[pos[i, 1], pos[j, 1]], mode="lines",
                                             line=dict(width=1 + 10 * jac[i, j], color="rgba(15,118,110,0.35)"),
                                             hoverinfo="text", text=f"{tags[i]} ↔ {tags[j]}: {jac[i, j]:.2f}",
                                             showlegend=False))
        fig.add_trace(go.Scatter(x=pos[:, 0], y=pos[:, 1], mode="markers+text", text=tags, textposition="top center",
                                 marker=dict(size=12 + 40 * freq / freq.max(), color=PRIMARY, line=dict(width=1, color="white")),
                                 hovertext=[f"{t}: {int(f)} posts" for t, f in zip(tags, freq)], hoverinfo="text",
                                 showlegend=False))
        fig.update_layout(template=PX_TEMPLATE, height=560, margin=dict(l=10, r=10, t=10, b=10),
                          xaxis=dict(visible=False, range=[-1.4, 1.4]), yaxis=dict(visible=False, range=[-1.4, 1.4], scaleanchor="x"))
        st.plotly_chart(fig, width="stretch")

        st.subheader("Strongest symptom pairs")
        by = st.radio("Rank by", ["lift", "phi", "jaccard"], horizontal=True)
        st.dataframe(nlp.top_pairs(metrics, by=by, k=15).round(2), width="stretch", hide_index=True)

        st.subheader("How does #fever co-occur?")
        if "#fever" in binary.columns:
            row = metrics["jaccard"].loc["#fever"].dropna().sort_values(ascending=False).head(8)
            fig = px.bar(row[::-1], orientation="h", template=PX_TEMPLATE, color_discrete_sequence=[PRIMARY],
                         labels={"value": "Jaccard with #fever", "index": ""})
            fig.update_layout(height=320, showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, width="stretch")

    # ---- Clusters -------------------------------------------------------------------------
    with tab2:
        c1, c2, c3 = st.columns(3)
        k = c1.slider("Number of clusters (k)", 3, 10, 6)
        method = c2.radio("Projection", ["PCA", "t-SNE"], horizontal=True)
        c3.caption("PCA keeps global structure. t-SNE separates local neighbourhoods; distances *between* clusters are not meaningful.")
        res = get_clusters(df, k)
        coords = get_projection(res.X, method.lower())
        plot_df = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1],
                                "cluster": [res.names[c] for c in res.labels],
                                "severity": df["triage_severity"].to_numpy(), "city": df["city"].to_numpy(),
                                "text": df["raw_text"].str.slice(0, 140).to_numpy(),
                                "hashtags": df["hashtags"].to_numpy()})
        color_by = st.radio("Colour by", ["cluster", "severity", "city"], horizontal=True)
        fig = px.scatter(plot_df, x="x", y="y", color=color_by, template=PX_TEMPLATE, opacity=0.75,
                         color_discrete_map=SEV_COLORS if color_by == "severity" else None,
                         category_orders={"severity": nlp.SEVERITY_ORDER},
                         hover_data={"x": False, "y": False, "text": True, "hashtags": True, "severity": True, "city": True})
        fig.update_traces(marker=dict(size=7))
        fig.update_layout(height=560, margin=dict(l=10, r=10, t=10, b=10), xaxis_title=f"{method} 1", yaxis_title=f"{method} 2")
        st.plotly_chart(fig, width="stretch")

        m1, m2 = st.columns(2)
        m1.metric("Silhouette score", f"{res.silhouette:.3f}", help="−1…1. Above ~0.2 on short text is acceptable.")
        m2.metric("Variance kept by LSA", f"{res.explained_variance:.0%}")

        st.subheader("Cluster profiles")
        prof = (df.assign(cluster=[res.names[c] for c in res.labels])
                .groupby("cluster").agg(posts=("post_id", "size"), avg_severity=("severity_num", "mean"),
                                        avg_urgency=("urgency_score", "mean"),
                                        top_city=("city", lambda s: s.value_counts().index[0]))
                .round(2).reset_index())
        prof["top terms"] = [", ".join(res.top_terms[c][:6]) for c in sorted(res.top_terms)]
        st.dataframe(prof, width="stretch", hide_index=True)

        with st.expander("Choosing k: elbow and silhouette sweep"):
            sweep = get_k_sweep(df)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=sweep["k"], y=sweep["inertia"], name="Inertia (elbow)", mode="lines+markers"))
            fig.add_trace(go.Scatter(x=sweep["k"], y=sweep["silhouette"], name="Silhouette", mode="lines+markers", yaxis="y2"))
            fig.update_layout(template=PX_TEMPLATE, height=340, xaxis_title="k", yaxis=dict(title="Inertia"),
                              yaxis2=dict(title="Silhouette", overlaying="y", side="right"), margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, width="stretch")


# =========================================================================================
# PAGE 3: Triage simulator
# =========================================================================================
EXAMPLES = {
    "High fever + joint pain": "High fever and joint pain for 2 days, also pain behind my eyes",
    "Migraine": "Severe throbbing headache on right side for 3 days, feeling dizzy and sensitive to light",
    "Food allergy": "Red itchy bumps on forearm after eating shrimp",
    "Chest pain (emergency)": "Crushing chest pain spreading to my left arm with cold sweat since this morning",
    "Anxiety": "Constant anxiety, racing thoughts and no sleep for days. Should I see a doctor?",
    "Negation test": "I have a cold but no fever and no chest pain, just a slightly blocked nose",
}


def page_triage() -> None:
    st.title("Real-Time NLP Patient Triage Simulator")
    st.warning("⚠️ Educational demo trained on **synthetic** data. It is not a medical device and gives no diagnosis. "
               "In an emergency call your local emergency number (India: 112).")
    model = get_triage_model(df)

    ex = st.selectbox("Load an example", ["(write your own)"] + list(EXAMPLES))
    text = st.text_area("Patient post", value=EXAMPLES.get(ex, ""), height=140,
                        placeholder="Describe symptoms, e.g. 'High fever and joint pain for 2 days'")
    go_btn = st.button("Analyse post", type="primary")

    if go_btn and text.strip():
        r = nlp.analyze_post(text, model)

        if r.crisis:
            st.markdown("<div class='crisis'><b>You are not alone.</b> If you are thinking of harming yourself, please reach out now: "
                        "<b>Tele-MANAS 14416</b> (India, free, 24×7) or call <b>112</b>. Talking to someone you trust helps too.</div>",
                        unsafe_allow_html=True)
        if not r.assessable:
            st.info(f"ℹ️ {r.notes[0]} Try describing symptoms in English, e.g. “fever and cough for 3 days”.")
            return

        left, right = st.columns([3, 2])
        with left:
            chips = "".join(f"<span class='chip'>{e}</span>" for e in r.entities) or "<i>none found</i>"
            tags = "".join(f"<span class='tag'>{t}</span>" for t in r.hashtags)
            st.markdown(
                f"""<div class='card'>
                <div style='color:#64748B;font-size:.85rem'>PREDICTED TRIAGE URGENCY</div>
                <span class='badge' style='background:{SEV_COLORS[r.severity]}'>{r.severity}</span>
                <span style='color:#64748B;margin-left:10px'>model confidence {r.confidence:.0%}</span>
                <hr style='border:none;border-top:1px solid #E2E8F0;margin:14px 0'>
                <div style='color:#64748B;font-size:.85rem'>RECOMMENDED DEPARTMENT</div>
                <div style='font-size:1.25rem;font-weight:600'>{r.department}</div>
                <div style='color:#64748B;font-size:.85rem;margin-top:12px'>EXTRACTED SYMPTOMS</div>{chips}
                <div style='color:#64748B;font-size:.85rem;margin-top:12px'>SUGGESTED HASHTAGS</div>{tags}
                </div>""", unsafe_allow_html=True)
            for n in r.notes:
                st.caption("↳ " + n)
            if r.possible_dengue:
                st.info("Fever together with joint pain / eye pain / rash is a pattern seen in dengue. Consider testing and seek care early.")
            if r.duration_days:
                st.caption(f"Detected symptom duration: ~{r.duration_days:g} day(s). Longer duration slightly raises the urgency score.")
        with right:
            gauge = go.Figure(go.Indicator(mode="gauge+number", value=r.urgency_score, title={"text": "Urgency score"},
                                           gauge=dict(axis=dict(range=[0, 100]), bar=dict(color="#334155"),
                                                      steps=[dict(range=[0, 25], color="#DCFCE7"), dict(range=[25, 50], color="#FEF9C3"),
                                                             dict(range=[50, 75], color="#FFEDD5"), dict(range=[75, 100], color="#FEE2E2")])))
            gauge.update_layout(height=230, margin=dict(l=20, r=20, t=40, b=0), template=PX_TEMPLATE)
            st.plotly_chart(gauge, width="stretch")
            pb = pd.DataFrame({"severity": list(r.probabilities), "probability": list(r.probabilities.values())})
            bar = px.bar(pb, x="probability", y="severity", orientation="h", color="severity", color_discrete_map=SEV_COLORS,
                         template=PX_TEMPLATE, category_orders={"severity": nlp.SEVERITY_ORDER[::-1]})
            bar.update_layout(height=220, showlegend=False, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(range=[0, 1], tickformat=".0%"))
            st.plotly_chart(bar, width="stretch")
            if r.urgency_terms:
                st.caption("Urgency keywords: " + ", ".join(r.urgency_terms))
            if r.red_flags:
                st.error("🚩 Red flags: " + "; ".join(r.red_flags))
    elif go_btn:
        st.info("Please enter some text first.")

    with st.expander("Model performance (hold-out set) and limitations"):
        m = model.metrics
        a, b, c = st.columns(3)
        a.metric("Accuracy", f"{m['accuracy']:.1%}")
        b.metric("Macro F1", f"{m['macro_f1']:.1%}")
        c.metric("Within one level", f"{m['within_one_level']:.1%}")
        st.dataframe(m["confusion"], width="stretch")
        st.caption("These numbers are measured on template-generated synthetic text, so they are optimistic and "
                   "do **not** reflect real-world accuracy. Safety rules (red-flag and explicit-severity floors) can only raise severity.")


{"Overview & Trends": page_overview, "Symptom Mining": page_mining, "Triage Simulator": page_triage}[page]()
