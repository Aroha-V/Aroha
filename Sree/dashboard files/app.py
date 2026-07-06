"""
India Weekly Disease Outbreak Dashboard
Map-driven analytical interface using Streamlit + Plotly
"""

import json
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="India Disease Outbreak Monitor",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    /* Dark medical theme */
    .stApp {
        background-color: #0a0e1a;
        color: #e2e8f0;
    }

    .main-header {
        font-family: 'Space Mono', monospace;
        font-size: 1.1rem;
        letter-spacing: 0.15em;
        color: #64ffda;
        text-transform: uppercase;
        margin-bottom: 0.2rem;
    }

    .sub-header {
        font-size: 0.85rem;
        color: #718096;
        letter-spacing: 0.05em;
        margin-bottom: 1.5rem;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #111827 0%, #1a2234 100%);
        border: 1px solid #1e3a5f;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.5rem;
    }

    .metric-value {
        font-family: 'Space Mono', monospace;
        font-size: 1.6rem;
        font-weight: 700;
        color: #64ffda;
        line-height: 1;
    }

    .metric-label {
        font-size: 0.72rem;
        color: #718096;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-top: 0.3rem;
    }

    /* Risk badges */
    .badge-high   { background:#ff4d4d22; color:#ff4d4d; border:1px solid #ff4d4d44; border-radius:4px; padding:2px 8px; font-size:0.75rem; font-family:'Space Mono',monospace; }
    .badge-medium { background:#ff900022; color:#ff9000; border:1px solid #ff900044; border-radius:4px; padding:2px 8px; font-size:0.75rem; font-family:'Space Mono',monospace; }
    .badge-low    { background:#00ff8822; color:#00ff88; border:1px solid #00ff8844; border-radius:4px; padding:2px 8px; font-size:0.75rem; font-family:'Space Mono',monospace; }
    .badge-none   { background:#44444422; color:#888;    border:1px solid #44444444; border-radius:4px; padding:2px 8px; font-size:0.75rem; font-family:'Space Mono',monospace; }

    /* Detail panel */
    .detail-panel {
        background: linear-gradient(135deg, #0d1526 0%, #111827 100%);
        border: 1px solid #1e3a5f;
        border-left: 3px solid #64ffda;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        margin-top: 1rem;
    }

    .detail-title {
        font-family: 'Space Mono', monospace;
        color: #64ffda;
        font-size: 1rem;
        margin-bottom: 0.8rem;
    }

    .insight-text {
        background: #0a0e1a;
        border-left: 2px solid #64ffda44;
        padding: 0.6rem 1rem;
        border-radius: 0 6px 6px 0;
        font-size: 0.88rem;
        color: #a0aec0;
        line-height: 1.6;
        margin-top: 0.8rem;
    }

    /* Selectbox styling */
    .stSelectbox > div > div {
        background-color: #111827;
        border: 1px solid #1e3a5f;
        color: #e2e8f0;
        border-radius: 6px;
    }

    /* Section divider */
    .section-divider {
        border: none;
        border-top: 1px solid #1e3a5f;
        margin: 1.5rem 0;
    }

    /* Hide default streamlit elements */
    #MainMenu, footer, header { visibility: hidden; }

    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# STATE NAME NORMALISER
# Maps CSV state names → GeoJSON feature names
# ─────────────────────────────────────────────
STATE_NAME_MAP = {
    "Andaman & Nicobar Islands": "Andaman and Nicobar",
    "Andaman And Nicobar Islands": "Andaman and Nicobar",
    "Andaman & Nicobar": "Andaman and Nicobar",
    "Dadra & Nagar Haveli": "Dadra and Nagar Haveli and Daman and Diu",
    "Dadra And Nagar Haveli": "Dadra and Nagar Haveli and Daman and Diu",
    "Dadra & Nagar Haveli And Daman & Diu": "Dadra and Nagar Haveli and Daman and Diu",
    "Daman & Diu": "Dadra and Nagar Haveli and Daman and Diu",
    "Jammu & Kashmir": "Jammu and Kashmir",
    "Jammu And Kashmir": "Jammu and Kashmir",
    "Orissa": "Odisha",
    "Uttaranchal": "Uttarakhand",
    "Delhi": "NCT of Delhi",
    "Nct Of Delhi": "NCT of Delhi",
    "Pondicherry": "Puducherry",
}


def normalise_state(name: str) -> str:
    name = str(name).strip().title()
    return STATE_NAME_MAP.get(name, name)


# ─────────────────────────────────────────────
# GEOJSON LOADER  (India states)
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_geojson():
    url = (
        "https://raw.githubusercontent.com/geohacker/india/master/state/india_state.geojson"
    )
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        geojson = r.json()
        sample = geojson["features"][0]["properties"] if geojson["features"] else {}
        return geojson, list(sample.keys())
    except Exception as e:
        st.error(f"⚠️ Could not load GeoJSON: {e}")
        return None, []


# ─────────────────────────────────────────────
# DATA LOADER & PROCESSOR
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_raw_data(path: str = "IDSP 2024 report  - Sheet1.csv") -> pd.DataFrame:
    """Load and clean the CSV without any aggregation (fallback for heatmap)."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df["Week"] = df["Week"].astype(str).str.extract(r"(\d+)").astype(int)
    df["No. of Cases"] = pd.to_numeric(df["No. of Cases"], errors="coerce").fillna(0)
    df["No. of Deaths"] = pd.to_numeric(df["No. of Deaths"], errors="coerce").fillna(0)
    df["State/UT"] = (
        df["State/UT"]
        .astype(str)
        .str.strip()
        .str.replace("\n", " ", regex=True)
        .str.title()
        .apply(normalise_state)
    )
    df["Disease/Illness"] = df["Disease/Illness"].astype(str).str.strip().str.title()
    return df


@st.cache_data(show_spinner=False)
def load_data(path: str = "IDSP 2024 report  - Sheet1.csv"):
    """Return two DataFrames: (agg, disease_weekly, raw_df)."""
    raw_df = load_raw_data(path)

    # ---- 1) Per‑state‑week total cases + top disease (for map & overview) ----
    top_disease = (
        raw_df.groupby(["Week", "State/UT", "Disease/Illness"])["No. of Cases"]
        .sum()
        .reset_index()
        .sort_values("No. of Cases", ascending=False)
        .drop_duplicates(subset=["Week", "State/UT"])
        .rename(columns={"Disease/Illness": "top_disease"})
    )
    agg_total = (
        raw_df.groupby(["Week", "State/UT"])["No. of Cases"]
        .sum()
        .reset_index()
    )
    agg = agg_total.merge(
        top_disease[["Week", "State/UT", "top_disease"]],
        on=["Week", "State/UT"],
        how="left"
    )
    agg["top_disease"] = agg["top_disease"].fillna("Unknown")

    # ---- 2) Per‑disease weekly data (for heatmap) ----
    disease_weekly = (
        raw_df.groupby(["Week", "State/UT", "Disease/Illness"])["No. of Cases"]
        .sum()
        .reset_index()
        .rename(columns={"Disease/Illness": "disease", "No. of Cases": "cases"})
    )

    return agg, disease_weekly, raw_df


def build_week_data(agg: pd.DataFrame, selected_week: int) -> pd.DataFrame:
    all_states = sorted(agg["State/UT"].unique())

    cur = agg[agg["Week"] == selected_week].copy().rename(columns={"No. of Cases": "cases"})
    prev = agg[agg["Week"] == selected_week - 1][["State/UT", "No. of Cases"]].rename(
        columns={"No. of Cases": "prev_cases"}
    )

    week_data = pd.DataFrame({"State/UT": all_states}).merge(cur, on="State/UT", how="left")
    week_data["cases"] = week_data["cases"].fillna(0)
    week_data["Week"] = selected_week
    week_data["top_disease"] = week_data["top_disease"].fillna("Unknown")
    week_data = week_data.merge(prev, on="State/UT", how="left")
    week_data["prev_cases"] = week_data["prev_cases"].fillna(0)

    week_data["growth_pct"] = week_data.apply(
        lambda r: ((r["cases"] - r["prev_cases"]) / r["prev_cases"] * 100)
        if r["prev_cases"] > 0
        else (100.0 if r["cases"] > 0 else 0.0),
        axis=1,
    ).round(1)

    def risk(r):
        if r["cases"] == 0:
            return "No Cases"
        if r["cases"] > 200 or r["growth_pct"] > 50:
            return "High"
        if r["cases"] > 50:
            return "Medium"
        return "Low"

    week_data["risk_level"] = week_data.apply(risk, axis=1)
    return week_data


# ─────────────────────────────────────────────
# CHOROPLETH MAP BUILDER
# ─────────────────────────────────────────────
def build_choropleth(week_data: pd.DataFrame, geojson: dict, id_field: str) -> go.Figure:
    severity_map = {"No Cases": 0, "Low": 1, "Medium": 2, "High": 3}
    week_data = week_data.copy()
    week_data["severity_num"] = week_data["risk_level"].map(severity_map).fillna(0)

    colorscale = [
        [0.00, "#1e3a5f"],
        [0.33, "#1e3a5f"],
        [0.34, "#00cc66"],
        [0.66, "#ff9000"],
        [0.67, "#ff9000"],
        [1.00, "#ff4d4d"],
    ]

    fig = go.Figure(
        go.Choropleth(
            geojson=geojson,
            locations=week_data["State/UT"],
            featureidkey=f"properties.{id_field}",
            z=week_data["severity_num"],
            colorscale=colorscale,
            zmin=0,
            zmax=3,
            customdata=np.stack([
                week_data["State/UT"],
                week_data["cases"].astype(int),
                week_data["growth_pct"],
                week_data["risk_level"],
                week_data["top_disease"],
            ], axis=-1),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Cases: <b>%{customdata[1]}</b><br>"
                "Growth: <b>%{customdata[2]}%</b><br>"
                "Risk: <b>%{customdata[3]}</b><br>"
                "Top Disease: <b>%{customdata[4]}</b>"
                "<extra></extra>"
            ),
            showscale=False,
            marker_line_color="#0a0e1a",
            marker_line_width=0.8,
        )
    )

    fig.update_geos(
        fitbounds="locations",
        visible=False,
        bgcolor="#0a0e1a",
    )

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="#0a0e1a",
        plot_bgcolor="#0a0e1a",
        geo=dict(bgcolor="#0a0e1a"),
        height=560,
        dragmode=False,
    )
    return fig


# ─────────────────────────────────────────────
# SEASONAL HEATMAP  (Disease × Week)
# ─────────────────────────────────────────────
DISEASE_COLORSCALES = {
    "Dengue":        [[0, "#fff0f0"], [1, "#b30000"]],
    "Malaria":       [[0, "#fff0f0"], [1, "#cc0033"]],
    "Typhoid":       [[0, "#fffbf0"], [1, "#b35900"]],
    "Leptospirosis": [[0, "#f0fff8"], [1, "#006666"]],
    "Cholera":       [[0, "#f0f8ff"], [1, "#1a4f8a"]],
}
DEFAULT_COLORSCALE = [[0, "#f5f5f5"], [1, "#333333"]]

MONTH_TICKS = {1:"Jan",5:"Feb",9:"Mar",14:"Apr",18:"May",22:"Jun",
               27:"Jul",31:"Aug",36:"Sep",40:"Oct",44:"Nov",49:"Dec"}


def build_state_heatmap(df, state, current_week):
    sdf = df[df["State/UT"] == state]
    pivot = (
        sdf.groupby(["Disease/Illness","Week"])["No. of Cases"]
        .sum().reset_index()
        .pivot(index="Disease/Illness", columns="Week", values="No. of Cases")
        .fillna(0)
    )
    for w in range(1, 53):
        if w not in pivot.columns: pivot[w] = 0
    pivot = pivot[sorted(pivot.columns)]
 
    z      = pivot.values
    z_log  = np.log1p(z)
 
    fig = go.Figure(go.Heatmap(
        z=z_log,
        x=list(pivot.columns),
        y=pivot.index.tolist(),
        colorscale=[
            [0.00,"#060e1c"],[0.15,"#0a2040"],
            [0.40,"#0d4f8a"],[0.65,"#1a7fd4"],
            [0.82,"#ff9800"],[0.93,"#ff5252"],
            [1.00,"#ff1744"],
        ],
        hovertemplate="<b>%{y}</b><br>Week %{x}<br>Cases: %{customdata:,}<extra></extra>",
        customdata=z,
        showscale=True,
        colorbar=dict(
            title=dict(text="log", font=dict(color="#3d6a9e", size=9)),
            tickfont=dict(color="#3d6a9e", size=9),
            bgcolor="#071630", bordercolor="#1b3d70",
            thickness=10, len=0.85, x=1.01,
        ),
        xgap=0.4, ygap=0.4,
    ))
 
    # NOW marker
    fig.add_vline(
        x=current_week,
        line=dict(color="#6fb3ff", width=1.5, dash="dot"),
        annotation_text="▼ NOW",
        annotation_font=dict(color="#6fb3ff", size=8, family="Space Mono"),
        annotation_position="top",
    )
 
    fig.update_layout(
        paper_bgcolor="#071630", plot_bgcolor="#071630",
        font=dict(color="#3d6a9e", size=10),
        margin=dict(l=0, r=40, t=22, b=10),
        height=300,
        xaxis=dict(
            title=dict(text="Week of Year", font=dict(color="#3d6a9e", size=10)),
            tickmode="linear", tick0=1, dtick=4,
            tickfont=dict(color="#3d6a9e", size=9),
            gridcolor="#0d2240", zeroline=False,
        ),
        yaxis=dict(
            tickfont=dict(color="#7aa8d0", size=9),
            gridcolor="#0d2240", zeroline=False,
        ),
    )
    return fig


from plotly.subplots import make_subplots

def build_disease_heatmap(disease_weekly: pd.DataFrame, state: str, current_week: int, raw_df=None) -> go.Figure:

    sdf = disease_weekly[disease_weekly["State/UT"] == state].copy()

    if sdf.empty and raw_df is not None:
        sdf = (
            raw_df[raw_df["State/UT"] == state]
            .groupby(["Disease/Illness", "Week"])["No. of Cases"]
            .sum()
            .reset_index()
            .rename(columns={"Disease/Illness": "disease", "No. of Cases": "cases"})
        )

    if sdf.empty:
        fig = go.Figure()
        fig.update_layout(
            paper_bgcolor="#071630", plot_bgcolor="#071630", height=220,
            annotations=[dict(text="No data available", x=0.5, y=0.5,
                              font=dict(color="#3d6a9e"), showarrow=False)]
        )
        return fig

    # ── Top 5 diseases by total cases ─────────────────────────────────────
    top5 = (
        sdf.groupby("disease")["cases"]
        .sum()
        .nlargest(5)
        .index.tolist()
    )
    sdf = sdf[sdf["disease"].isin(top5)]

    # ── Build pivot (all 52 weeks guaranteed) ─────────────────────────────
    pivot = (
        sdf.pivot_table(index="disease", columns="Week", values="cases", aggfunc="sum")
        .fillna(0)
    )
    for w in range(1, 53):
        if w not in pivot.columns:
            pivot[w] = 0
    pivot = pivot[sorted(pivot.columns)]
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]

    diseases = pivot.index.tolist()
    weeks    = list(range(1, 53))   # always 1‑52
    n        = len(diseases)

    DISEASE_COLORS = {
        "Dengue":                  ["#fff0f0", "#ffb3b3", "#ff6666", "#cc0000", "#7a0000"],
        "Malaria":                 ["#fff0f0", "#ffb3b3", "#ff6666", "#cc0000", "#7a0000"],
        "Typhoid":                 ["#fffaf0", "#ffdfa0", "#ffb732", "#d4810a", "#7a4600"],
        "Leptospirosis":           ["#f0fffc", "#b3ede0", "#40c9a2", "#007f6e", "#004d42"],
        "Cholera":                 ["#f0f8ff", "#a8d4f5", "#4a9fd4", "#1a5fa8", "#0a3060"],
        "Acute Diarrheal Disease": ["#fff0f8", "#ffb3d9", "#ff66aa", "#cc0066", "#7a003d"],
        "Food Poisoning":          ["#f5fff0", "#b3f0b3", "#44cc44", "#228b22", "#0a5214"],
        "Scrub Typhus":            ["#fff8f0", "#ffd4a8", "#ffaa55", "#cc6600", "#7a3d00"],
        "Japanese Encephalitis":   ["#f8f0ff", "#d4a8ff", "#aa55ff", "#6600cc", "#3d007a"],
        "Measles & Rubella":       ["#fffff0", "#f0e68c", "#daa520", "#b8860b", "#6b4c00"],
        "Chickenpox":              ["#f0f8ff", "#a8c8f0", "#5599dd", "#2255aa", "#0a2d6b"],
        "Diphtheria":              ["#fff0f0", "#ffccaa", "#ff8844", "#cc4400", "#7a2900"],
        "Fever Of Unknown Origin": ["#f0fff8", "#a8f0cc", "#33cc88", "#008855", "#004d33"],
        "Only Fever <7 Days":      ["#fff8f0", "#ffe0b3", "#ffaa33", "#cc7700", "#7a4400"],
        "Fever With Rash":         ["#fff0ff", "#f0b3f0", "#cc44cc", "#880088", "#4d004d"],
        "Fever":                   ["#fff8f0", "#ffe0b3", "#ffaa33", "#cc7700", "#7a4400"],
        "Animal Bite - Dog Bite":  ["#f0fff8", "#a8f0d4", "#33ccaa", "#008877", "#004d44"],
    }
    DEFAULT_COLORS = ["#f5f5ff", "#b3b3ee", "#6666cc", "#222299", "#0a0a5c"]

    week_tickvals = list(range(1, 53, 4))
    week_ticktext = [str(w) for w in week_tickvals]

    from plotly.subplots import make_subplots

    cell_px        = 14
    row_gap_px     = 6
    axis_margin_px = 70
    fig_height     = n * cell_px + (n - 1) * row_gap_px + axis_margin_px

    fig = make_subplots(
        rows=n, cols=1,
        shared_xaxes=True,
        vertical_spacing=row_gap_px / max(fig_height, 1),
        row_heights=[1.0] * n,
    )

    for i, disease in enumerate(diseases):
        row_vals  = pivot.loc[disease].reindex(range(1, 53), fill_value=0).values.astype(float)
        row_max   = row_vals.max()
        norm_vals = row_vals / row_max if row_max > 0 else row_vals

        colors = DISEASE_COLORS.get(disease, DEFAULT_COLORS)
        cscale = [[j / (len(colors) - 1), c] for j, c in enumerate(colors)]

        fig.add_trace(
            go.Heatmap(
                z=[norm_vals],
                x=weeks,
                y=[disease],
                colorscale=cscale,
                zmin=0, zmax=1,
                showscale=False,
                text=[[int(v) for v in row_vals]],
                hovertemplate=(
                    "<b>" + disease + "</b><br>"
                    "Week %{x}<br>"
                    "Cases: %{text:,}"
                    "<extra></extra>"
                ),
                xgap=2,
                ygap=0,
            ),
            row=i + 1, col=1,
        )

        fig.update_yaxes(
            tickvals=[disease],
            ticktext=[disease],
            tickfont=dict(color="#9ab8d4", size=8.5),
            showgrid=False,
            zeroline=False,
            row=i + 1, col=1,
        )
        fig.update_xaxes(
            showgrid=False,
            zeroline=False,
            row=i + 1, col=1,
        )

    # ── Bottom x-axis: weeks 1–52 ─────────────────────────────────────────
    fig.update_xaxes(
        range=[0.5, 52.5],          # ← show all 52 weeks, no clipping
        tickmode="array",
        tickvals=week_tickvals,
        ticktext=week_ticktext,
        tickfont=dict(color="#7aa8d0", size=9),
        title=dict(text="Week", font=dict(color="#3d6a9e", size=9)),
        row=n, col=1,
    )

    # ── Current week highlight: solid bright vertical band ────────────────
    fig.add_vrect(
        x0=current_week - 0.5,
        x1=current_week + 0.5,
        fillcolor="rgba(100, 255, 218, 0.18)",   # teal glow
        line=dict(color="#64ffda", width=1.5),
        layer="above",
    )

    # Label above the highlight
    fig.add_annotation(
        x=current_week,
        y=1.0,
        yref="paper",
        text=f"▼ W{current_week}",
        showarrow=False,
        font=dict(color="#64ffda", size=8, family="Space Mono"),
        xanchor="center",
        yanchor="bottom",
    )

    fig.update_layout(
        paper_bgcolor="#071630",
        plot_bgcolor="#071630",
        margin=dict(l=10, r=20, t=24, b=40),
        height=fig_height,
        font=dict(color="#3d6a9e", size=9),
        hovermode="closest",
    )

    return fig
# ─────────────────────────────────────────────
# INSIGHT TEXT GENERATOR
# ─────────────────────────────────────────────
def generate_insight(row: pd.Series) -> str:
    state = row["State/UT"]
    cases = int(row["cases"])
    growth = row["growth_pct"]
    disease = row["top_disease"]
    risk = row["risk_level"]

    if cases == 0:
        return f"No reported cases in {state} this week. Situation appears stable."

    direction = "increase" if growth >= 0 else "decrease"
    arrow = "📈" if growth > 10 else ("📉" if growth < -10 else "➡️")

    lines = [
        f"{arrow} {abs(growth):.1f}% week-over-week {direction} in {state}.",
        f"Primary driver: <b>{disease}</b> accounting for the majority of reported cases.",
    ]

    if risk == "High":
        lines.append("⚠️ Risk level is HIGH — immediate surveillance and response recommended.")
    elif risk == "Medium":
        lines.append("🟠 Moderate activity — continued monitoring advised.")
    else:
        lines.append("🟢 Situation under control at current levels.")

    return " ".join(lines)


# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────
def main():
    st.markdown('<p class="main-header">🦠 India Disease Outbreak Monitor</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">IDSP Surveillance · Weekly Analysis Interface</p>', unsafe_allow_html=True)

    # ── Load data ───────────────────────────
    with st.spinner("Loading surveillance data..."):
        try:
            agg, disease_weekly, raw_df = load_data()
        except FileNotFoundError:
            st.error("CSV file not found. Place `IDSP 2024 report  - Sheet1.csv` in the same directory.")
            st.stop()
        except Exception as e:
            st.error(f"Error loading data: {e}")
            st.stop()

    geojson, prop_keys = load_geojson()
    if geojson is None:
        st.stop()

    name_candidates = ["NAME_1", "st_nm", "name", "NAME", "State", "state"]
    id_field = next((k for k in name_candidates if k in prop_keys), prop_keys[0] if prop_keys else "NAME_1")

    # ── Week selector row ───────────────────
    weeks = sorted(agg["Week"].unique())
    col_sel, col_legend = st.columns([3, 7])

    with col_sel:
        selected_week = st.selectbox(
            "Select Epidemiological Week",
            weeks,
            index=len(weeks) - 1,
            label_visibility="visible",
        )

    with col_legend:
        st.markdown(
            """
            <div style="display:flex;align-items:center;gap:12px;margin-top:28px;flex-wrap:wrap;">
                <span style="font-size:0.75rem;color:#718096;font-family:'Space Mono',monospace;">RISK LEGEND</span>
                <span class="badge-high">HIGH</span>
                <span class="badge-medium">MEDIUM</span>
                <span class="badge-low">LOW</span>
                <span class="badge-none">NO CASES</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Build current week data ─────────────
    week_data = build_week_data(agg, selected_week)

    # ── KPI row ─────────────────────────────
    total_cases = int(week_data["cases"].sum())
    high_risk_n = int((week_data["risk_level"] == "High").sum())
    growing = int((week_data["growth_pct"] > 20).sum())
    top_state = week_data.loc[week_data["cases"].idxmax(), "State/UT"] if total_cases > 0 else "—"

    k1, k2, k3, k4 = st.columns(4)
    for col, val, label in [
        (k1, f"{total_cases:,}", "TOTAL CASES"),
        (k2, str(high_risk_n), "HIGH RISK STATES"),
        (k3, str(growing), "RAPID GROWTH (>20%)"),
        (k4, top_state, "HIGHEST BURDEN STATE"),
    ]:
        col.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value">{val}</div>'
            f'<div class="metric-label">{label}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Map + State detail columns ───────────
    map_col, detail_col = st.columns([6, 4], gap="large")

    with map_col:
        st.markdown(
            f'<p style="font-family:\'Space Mono\',monospace;font-size:0.75rem;'
            f'color:#718096;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:4px;">'
            f'CHOROPLETH · WEEK {selected_week} · CLICK A STATE TO ANALYSE</p>',
            unsafe_allow_html=True,
        )
        fig_map = build_choropleth(week_data, geojson, id_field)
        click_event = st.plotly_chart(
            fig_map,
            use_container_width=True,
            on_select="rerun",
            selection_mode="points",
            key="india_map",
        )

    # ── Extract clicked state ────────────────
    clicked_state = None
    if click_event and hasattr(click_event, "selection") and click_event.selection:
        pts = click_event.selection.get("points", [])
        if pts:
            loc = pts[0].get("location")
            if loc:
                clicked_state = loc

    with detail_col:
        st.markdown(
            '<p style="font-family:\'Space Mono\',monospace;font-size:0.75rem;'
            'color:#718096;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:4px;">'
            'STATE DETAIL PANEL</p>',
            unsafe_allow_html=True,
        )

        state_list = sorted(week_data["State/UT"].unique())
        default_idx = state_list.index(clicked_state) if clicked_state in state_list else 0
        selected_state = st.selectbox(
            "Select / search state",
            state_list,
            index=default_idx,
            label_visibility="collapsed",
        )

        # ── State metrics ────────────────────
        row = week_data[week_data["State/UT"] == selected_state].iloc[0]
        cases_val = int(row["cases"])
        growth_val = row["growth_pct"]
        risk_val = row["risk_level"]
        disease_val = row["top_disease"]

        badge_class = {
            "High": "badge-high",
            "Medium": "badge-medium",
            "Low": "badge-low",
            "No Cases": "badge-none",
        }.get(risk_val, "badge-none")

        growth_color = "#ff4d4d" if growth_val > 0 else "#00cc66"
        growth_sign = "+" if growth_val > 0 else ""

        st.markdown(
            f"""
            <div class="detail-panel">
                <div class="detail-title">{selected_state}</div>
                <div style="display:flex;gap:10px;align-items:center;margin-bottom:12px;">
                    <span class="{badge_class}">{risk_val}</span>
                    <span style="font-size:0.8rem;color:#718096;">Top: <b style="color:#e2e8f0">{disease_val}</b></span>
                </div>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">
                    <div>
                        <div style="font-family:'Space Mono',monospace;font-size:1.4rem;color:#64ffda;">{cases_val:,}</div>
                        <div style="font-size:0.7rem;color:#718096;text-transform:uppercase;letter-spacing:0.08em;">This Week</div>
                    </div>
                    <div>
                        <div style="font-family:'Space Mono',monospace;font-size:1.4rem;color:{growth_color};">{growth_sign}{growth_val:.1f}%</div>
                        <div style="font-size:0.7rem;color:#718096;text-transform:uppercase;letter-spacing:0.08em;">WoW Growth</div>
                    </div>
                </div>
                <div class="insight-text">{generate_insight(row)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Seasonal disease heatmap ─────────
        st.markdown(
            '<p style="font-family:\'Space Mono\',monospace;font-size:0.7rem;color:#718096;'
            'letter-spacing:0.08em;text-transform:uppercase;margin:12px 0 4px 0;">'
            'SEASONAL PATTERN · DISEASE × WEEK · DARKER = MORE CASES</p>',
            unsafe_allow_html=True,
        )
        heatmap_fig = build_disease_heatmap(disease_weekly, selected_state, selected_week, raw_df=raw_df)
        st.plotly_chart(heatmap_fig, use_container_width=True, key="heatmap_chart")

    # ── State comparison table ───────────────
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown(
        '<p style="font-family:\'Space Mono\',monospace;font-size:0.75rem;color:#718096;'
        'letter-spacing:0.1em;text-transform:uppercase;margin-bottom:8px;">ALL STATES · WEEK SUMMARY</p>',
        unsafe_allow_html=True,
    )

    display_df = (
        week_data[["State/UT", "cases", "prev_cases", "growth_pct", "risk_level", "top_disease"]]
        .rename(columns={
            "State/UT": "State / UT",
            "cases": "Cases",
            "prev_cases": "Prev Week",
            "growth_pct": "Growth %",
            "risk_level": "Risk",
            "top_disease": "Top Disease",
        })
        .sort_values("Cases", ascending=False)
    )

    risk_colors = {"High": "🔴", "Medium": "🟠", "Low": "🟢", "No Cases": "⚫"}
    display_df["Risk"] = display_df["Risk"].map(lambda x: f"{risk_colors.get(x,'')} {x}")
    display_df["Growth %"] = display_df["Growth %"].map(lambda x: f"+{x:.1f}%" if x >= 0 else f"{x:.1f}%")
    display_df["Cases"] = display_df["Cases"].astype(int)
    display_df["Prev Week"] = display_df["Prev Week"].astype(int)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=300,
    )


if __name__ == "__main__":
    main()
