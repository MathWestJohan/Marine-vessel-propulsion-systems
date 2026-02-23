import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from models.DigitalTwin import detect_faults
from dashboard.components import (
    BG, CARD, BORDER, TEXT, CYAN, TEAL, AMBER, RED, BLUE,
)

PLOT_BASE = dict(
    paper_bgcolor=BG,
    plot_bgcolor=CARD,
    font=dict(color=TEXT, family="Inter, system-ui, sans-serif", size=12),
    margin=dict(l=50, r=20, t=40, b=35),
    xaxis=dict(gridcolor=BORDER, zeroline=False),
    yaxis=dict(gridcolor=BORDER, zeroline=False),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    hovermode="x unified",
)


def _lay(**kw):
    d = {**PLOT_BASE}
    d.update(kw)
    return d


def chart_decay_trend(df, maintenance_history=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["comp_decay"], name="Compressor Decay",
        line=dict(color=CYAN, width=2), fill="tozeroy",
        fillcolor="rgba(6,182,212,0.06)"))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["turb_decay"], name="Turbine Decay",
        line=dict(color=AMBER, width=2), fill="tozeroy",
        fillcolor="rgba(245,158,11,0.06)"))
    
    # Add maintenance event markers
    if maintenance_history:
        for event in maintenance_history:
            idx = event['sample_index']
            if idx in df.index:
                fig.add_vline(x=idx, line_dash="dot", line_color=TEAL, line_width=1.5)
                fig.add_annotation(x=idx, y=1.01, text="Maint", showarrow=False, 
                                 font=dict(color=TEAL, size=9), textangle=-90)

    fig.add_hline(y=0.975, line_dash="dash", line_color=TEAL,
                  annotation_text="Maintenance Limit", annotation_font_color=TEAL,
                  annotation_font_size=10)
    y_min = min(df["comp_decay"].min(), df["turb_decay"].min()) - 0.01
    y_max = max(df["comp_decay"].max(), df["turb_decay"].max()) + 0.01
    fig.update_layout(**_lay(title="Degradation Trend", height=340,
                             yaxis=dict(gridcolor=BORDER, title="Coefficient",
                                        range=[y_min, y_max])))
    fig.update_xaxes(title_text="Sample", gridcolor=BORDER)
    return fig


def chart_innovation(df):
    _, metrics = detect_faults(df)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=["Compressor Innovations", "Turbine Innovations"])
    for i, label in enumerate(["Compressor", "Turbine"], 1):
        innov = metrics[label]["innovations"]
        sigma = metrics[label]["std_innov"]
        color = CYAN if i == 1 else AMBER
        fig.add_trace(go.Scatter(
            x=list(range(len(innov))), y=innov, name=f"{label} Innovation",
            line=dict(color=color, width=1), fill="tozeroy",
            fillcolor=f"rgba({','.join(str(int(color.lstrip('#')[j:j+2], 16)) for j in (0, 2, 4))},0.06)"),
            row=i, col=1)
        fig.add_hline(y=2.5 * sigma, line_dash="dot", line_color=RED,
                      annotation_text="+2.5\u03c3", annotation_font_color=RED,
                      annotation_font_size=9, row=i, col=1)
        fig.add_hline(y=-2.5 * sigma, line_dash="dot", line_color=RED,
                      annotation_text="-2.5\u03c3", annotation_font_color=RED,
                      annotation_font_size=9, row=i, col=1)
    fig.update_layout(paper_bgcolor=BG, plot_bgcolor=CARD,
                      font=dict(color=TEXT, size=11), height=400,
                      margin=dict(l=50, r=20, t=40, b=30),
                      legend=dict(bgcolor="rgba(0,0,0,0)"), hovermode="x unified",
                      showlegend=False)
    fig.update_xaxes(gridcolor=BORDER)
    fig.update_yaxes(gridcolor=BORDER)
    return fig


def chart_fuel_efficiency(df):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=df.index, y=df["fuel_flow"], name="Fuel Flow (kg/s)",
        line=dict(color=AMBER, width=2), fill="tozeroy",
        fillcolor="rgba(245,158,11,0.06)"), secondary_y=False)
    sfc = df["fuel_flow"] / (df["gt_torque"] / 1000).replace(0, np.nan)
    fig.add_trace(go.Scatter(
        x=df.index, y=sfc, name="SFC (kg/s per MW)",
        line=dict(color=CYAN, width=1.5, dash="dot")), secondary_y=True)
    fig.update_layout(**_lay(title="Fuel Consumption Trend", height=340))
    fig.update_yaxes(title_text="Fuel Flow (kg/s)", gridcolor=BORDER, secondary_y=False)
    fig.update_yaxes(title_text="SFC", gridcolor=BORDER, secondary_y=True)
    fig.update_xaxes(title_text="Sample", gridcolor=BORDER)
    return fig


def chart_temperatures(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["t48"], name="HP Turbine Exit (T48)",
                             line=dict(color=RED, width=2)))
    fig.add_trace(go.Scatter(x=df.index, y=df["t2"], name="Compressor Outlet (T2)",
                             line=dict(color=AMBER, width=2)))
    fig.add_trace(go.Scatter(x=df.index, y=df["t1"], name="Compressor Inlet (T1)",
                             line=dict(color=TEAL, width=2)))
    fig.update_layout(**_lay(title="Temperature Profiles (\u00b0C)", height=340,
                             yaxis=dict(gridcolor=BORDER, title="\u00b0C")))
    fig.update_xaxes(title_text="Sample", gridcolor=BORDER)
    return fig


def chart_pressures(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["p2"], name="Compressor Outlet (P2)",
                             line=dict(color=CYAN, width=2)))
    fig.add_trace(go.Scatter(x=df.index, y=df["p48"], name="HP Turbine Exit (P48)",
                             line=dict(color=BLUE, width=2)))
    fig.add_trace(go.Scatter(x=df.index, y=df["p1"], name="Compressor Inlet (P1)",
                             line=dict(color=TEAL, width=1.5, dash="dot")))
    fig.add_trace(go.Scatter(x=df.index, y=df["pexh"], name="Exhaust (Pexh)",
                             line=dict(color=AMBER, width=1.5, dash="dot")))
    fig.update_layout(**_lay(title="Pressure Profiles (bar)", height=340,
                             yaxis=dict(gridcolor=BORDER, title="bar")))
    fig.update_xaxes(title_text="Sample", gridcolor=BORDER)
    return fig


def chart_propulsion(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["ts"], name="Starboard (Ts)",
                             line=dict(color=CYAN, width=2)))
    fig.add_trace(go.Scatter(x=df.index, y=df["tp"], name="Port (Tp)",
                             line=dict(color=TEAL, width=2)))
    fig.add_trace(go.Scatter(x=df.index, y=abs(df["ts"] - df["tp"]),
                             name="|Ts - Tp| Imbalance",
                             line=dict(color=RED, width=1.5, dash="dot")))
    fig.update_layout(**_lay(title="Propeller Torque (kN)", height=340,
                             yaxis=dict(gridcolor=BORDER, title="kN")))
    fig.update_xaxes(title_text="Sample", gridcolor=BORDER)
    return fig


def make_gauge(value, title, min_val=0.85, max_val=1.0):
    if value >= 0.95:
        bar_color = TEAL
    elif value >= 0.90:
        bar_color = AMBER
    else:
        bar_color = RED
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number=dict(font=dict(size=36, color=TEXT), valueformat=".4f"),
        title=dict(text=title, font=dict(size=13, color=TEXT)),
        gauge=dict(
            axis=dict(range=[min_val, max_val], tickcolor=TEXT,
                      tickfont=dict(size=10, color=TEXT)),
            bar=dict(color=bar_color, thickness=0.55),
            bgcolor=BORDER,
            bordercolor=BORDER,
            steps=[
                dict(range=[min_val, 0.90], color="rgba(239,68,68,0.12)"),
                dict(range=[0.90, 0.95], color="rgba(245,158,11,0.12)"),
                dict(range=[0.95, max_val], color="rgba(45,212,191,0.12)"),
            ],
            threshold=dict(line=dict(color=RED, width=2), thickness=0.8, value=0.90),
        ),
    ))
    fig.update_layout(paper_bgcolor=BG, plot_bgcolor=CARD, font=dict(color=TEXT),
                      margin=dict(l=30, r=30, t=55, b=15), height=260)
    return fig