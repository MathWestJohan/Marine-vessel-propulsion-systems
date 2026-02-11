import pandas as pd
import numpy as np
import gradio as gr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os


class PropulsionDigitalTwin:
    """
    A Digital Twin for Marine Propulsion Systems.
    Integrates predictive models with health monitoring and maintenance alerts.
    """

    def __init__(self, compressor_model, turbine_model, scaler=None):
        self.compressor_model = compressor_model
        self.turbine_model = turbine_model
        self.scaler = scaler
        self.health_history = []
        self.MAINTENANCE_THRESHOLD = 0.96

    def predict_health(self, input_data):
        """
        Predicts health from actual sensor readings.
        input_data: DataFrame with feature columns matching training data
        """
        if self.scaler:
            input_scaled = self.scaler.transform(input_data)
        else:
            input_scaled = input_data.values
            
        comp_decay = self.compressor_model.predict(input_scaled)[0]
        turb_decay = self.turbine_model.predict(input_scaled)[0]
        
        status = {
            "compressor_health": round(comp_decay, 4),
            "turbine_health": round(turb_decay, 4),
            "comp_alert": comp_decay < self.MAINTENANCE_THRESHOLD,
            "turb_alert": turb_decay < self.MAINTENANCE_THRESHOLD
        }
        
        self.health_history.append(status)
        return status

    def get_maintenance_recommendation(self, status):
        """Logic for predictive maintenance."""
        recommendations = []
        if status["comp_alert"]:
            recommendations.append("⚠️ COMPRESSOR MAINTENANCE REQUIRED - Efficiency below 96%")
            recommendations.append("   → Schedule compressor inspection")
            recommendations.append("   → Check for fouling or erosion")
        if status["turb_alert"]:
            recommendations.append("⚠️ TURBINE MAINTENANCE REQUIRED - Efficiency below 96%")
            recommendations.append("   → Schedule turbine blade inspection")
            recommendations.append("   → Check for thermal degradation")

        if not recommendations:
            recommendations.append("✅ All systems operating within normal parameters")
        return "\n".join(recommendations)



# Dashboard Styling and Implementation

BG = "#0b1120"
CARD = "#111927"
BORDER = "#1e293b"
TEXT = "#cbd5e1"
DIM = "#64748b"
CYAN = "#06b6d4"
TEAL = "#2dd4bf"
AMBER = "#f59e0b"
RED = "#ef4444"
BLUE = "#3b82f6"

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


def compute_kalman_metrics(df):
    """Compute pseudo-Kalman innovation metrics from the dataset."""
    metrics = {}
    for col, label in [("comp_decay", "Compressor"), ("turb_decay", "Turbine")]:
        vals = df[col].values
        window = min(50, max(5, len(vals) // 20))
        running_mean = pd.Series(vals).rolling(window, min_periods=1).mean().values
        innovations = vals - running_mean
        metrics[label] = {
            "innovations": innovations,
            "mean_innov": float(np.mean(innovations)),
            "std_innov": float(np.std(innovations)),
            "covariance": float(np.var(innovations)),
            "max_abs": float(np.max(np.abs(innovations))),
        }
    return metrics


def detect_faults(df, threshold_sigma=2.5):
    """Flag samples where innovation exceeds threshold."""
    metrics = compute_kalman_metrics(df)
    flags = []
    for label in ["Compressor", "Turbine"]:
        m = metrics[label]
        innov = m["innovations"]
        sigma = m["std_innov"] if m["std_innov"] > 0 else 1e-9
        fault_mask = np.abs(innov) > threshold_sigma * sigma
        n_faults = int(np.sum(fault_mask))
        pct = 100 * n_faults / len(innov) if len(innov) > 0 else 0
        flags.append({"component": label, "n_faults": n_faults, "pct": pct,
                       "sigma": sigma, "threshold": threshold_sigma})
    return flags, metrics


def _card(title, value, unit, color, extra=""):
    return f"""
    <div style="background:{CARD};border:1px solid {BORDER};border-radius:8px;padding:14px 16px;min-width:0;">
        <div style="font-size:10px;color:{DIM};text-transform:uppercase;letter-spacing:0.7px;margin-bottom:4px;">{title}</div>
        <div style="font-size:24px;font-weight:700;color:{color};font-family:'JetBrains Mono',monospace;line-height:1.2;">{value}</div>
        <div style="font-size:10px;color:{DIM};margin-top:2px;">{unit}{extra}</div>
    </div>"""


def _badge(label, color):
    return f'<span style="display:inline-block;padding:2px 8px;border-radius:4px;font-size:10px;font-weight:700;background:{color}22;color:{color};border:1px solid {color}44;">{label}</span>'


def _status_color(val, warn=0.95, crit=0.90):
    if val >= warn:
        return TEAL, "HEALTHY"
    if val >= crit:
        return AMBER, "WARNING"
    return RED, "CRITICAL"


def sensor_html(df):
    """Display 14 measured parameters"""
    latest = df.iloc[-1]
    sensors = [
        ("GT Shaft Torque", f'{latest["gt_torque"]:,.1f}', "kN m", CYAN),
        ("GT Revolutions", f'{latest["gt_rpm"]:,.0f}', "rpm", CYAN),
        ("Gas Gen. Revolutions", f'{latest["gg_rpm"]:,.0f}', "rpm", CYAN),
        ("Starboard Prop. Torque", f'{latest["ts"]:,.1f}', "kN", BLUE),
        ("Port Prop. Torque", f'{latest["tp"]:,.1f}', "kN", BLUE),
        ("HP Turbine Exit Temp", f'{latest["t48"]:.1f}', "\u00b0C", RED),
        ("Compressor Inlet Temp", f'{latest["t1"]:.1f}', "\u00b0C", AMBER),
        ("Compressor Outlet Temp", f'{latest["t2"]:.1f}', "\u00b0C", AMBER),
        ("HP Turbine Exit Press.", f'{latest["p48"]:.3f}', "bar", TEAL),
        ("Compressor Inlet Press.", f'{latest["p1"]:.3f}', "bar", TEAL),
        ("Compressor Outlet Press.", f'{latest["p2"]:.3f}', "bar", TEAL),
        ("Exhaust Gas Pressure", f'{latest["pexh"]:.3f}', "bar", TEAL),
        ("Fuel Flow", f'{latest["fuel_flow"]:.4f}', "kg/s", AMBER),
        ("Ship Speed (response)", f'{latest["ship_speed"]:.0f}', "knots", CYAN),
    ]
    html = '<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:8px;">'
    for t, v, u, c in sensors:
        html += _card(t, v, u, c)
    html += '</div>'
    return html


def health_html(df):
    """Health status cards with Kalman metrics"""
    comp_val = df["comp_decay"].mean()
    turb_val = df["turb_decay"].mean()
    cc, cl = _status_color(comp_val)
    tc, tl = _status_color(turb_val)

    faults, metrics = detect_faults(df)
    comp_f = faults[0]
    turb_f = faults[1]

    # Maintenance estimation
    comp_trend = np.polyfit(range(len(df)), df["comp_decay"].values, 1)
    turb_trend = np.polyfit(range(len(df)), df["turb_decay"].values, 1)

    def est_remaining(slope, current, threshold=0.90):
        if slope >= 0:
            return "Stable"
        remaining = (threshold - current) / slope
        return f"{abs(remaining):,.0f} samples"

    comp_maint = est_remaining(comp_trend[0], comp_val)
    turb_maint = est_remaining(turb_trend[0], turb_val)

    html = f"""
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">
        <div style="background:{CARD};border:1px solid {BORDER};border-radius:8px;padding:16px;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
                <span style="font-size:13px;font-weight:600;color:{TEXT};">Compressor Health</span>
                {_badge(cl, cc)}
            </div>
            <div style="font-size:36px;font-weight:700;color:{cc};font-family:'JetBrains Mono',monospace;">{comp_val:.4f}</div>
            <div style="margin-top:10px;display:grid;grid-template-columns:1fr 1fr;gap:8px;font-size:11px;color:{DIM};">
                <div>Innovation &sigma;: <span style="color:{TEXT};">{metrics['Compressor']['std_innov']:.6f}</span></div>
                <div>Innovation Cov: <span style="color:{TEXT};">{metrics['Compressor']['covariance']:.2e}</span></div>
                <div>Fault Flags: <span style="color:{RED if comp_f['n_faults'] > 0 else TEAL};">{comp_f['n_faults']} ({comp_f['pct']:.1f}%)</span></div>
                <div>Est. Maintenance: <span style="color:{AMBER};">{comp_maint}</span></div>
            </div>
        </div>
        <div style="background:{CARD};border:1px solid {BORDER};border-radius:8px;padding:16px;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
                <span style="font-size:13px;font-weight:600;color:{TEXT};">Turbine Health</span>
                {_badge(tl, tc)}
            </div>
            <div style="font-size:36px;font-weight:700;color:{tc};font-family:'JetBrains Mono',monospace;">{turb_val:.4f}</div>
            <div style="margin-top:10px;display:grid;grid-template-columns:1fr 1fr;gap:8px;font-size:11px;color:{DIM};">
                <div>Innovation &sigma;: <span style="color:{TEXT};">{metrics['Turbine']['std_innov']:.6f}</span></div>
                <div>Innovation Cov: <span style="color:{TEXT};">{metrics['Turbine']['covariance']:.2e}</span></div>
                <div>Fault Flags: <span style="color:{RED if turb_f['n_faults'] > 0 else TEAL};">{turb_f['n_faults']} ({turb_f['pct']:.1f}%)</span></div>
                <div>Est. Maintenance: <span style="color:{AMBER};">{turb_maint}</span></div>
            </div>
        </div>
    </div>"""
    return html


def efficiency_html(df, baselines):
    """Operational efficiency HTML"""
    cur_fuel = df["fuel_flow"].mean()
    cur_torque = df["gt_torque"].mean()
    cur_speed = df["ship_speed"].mean()

    nearest_speed = baselines["ship_speed"].unique()
    nearest = min(nearest_speed, key=lambda x: abs(x - cur_speed))
    baseline_rows = baselines[baselines["ship_speed"] == nearest]
    
    if len(baseline_rows) > 0:
        b = baseline_rows.iloc[0]
        base_fuel = b["fuel_flow"]
        base_torque = b["gt_torque"]
        fuel_change = ((cur_fuel - base_fuel) / base_fuel * 100) if base_fuel > 0 else 0
        torque_change = ((cur_torque - base_torque) / base_torque * 100) if base_torque > 0 else 0
        sfc = cur_fuel / (cur_torque / 1000) if cur_torque > 0 else 0
        base_sfc = base_fuel / (base_torque / 1000) if base_torque > 0 else 0
        eff_change = ((sfc - base_sfc) / base_sfc * 100) if base_sfc > 0 else 0
    else:
        fuel_change = 0
        torque_change = 0
        eff_change = 0
        sfc = 0

    comp_deg = (1 - df["comp_decay"].mean()) * 100
    turb_deg = (1 - df["turb_decay"].mean()) * 100

    fc_color = TEAL if fuel_change <= 0 else RED
    tc_color = TEAL if torque_change >= 0 else RED
    ec_color = TEAL if eff_change <= 0 else RED

    html = f"""
    <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:8px;">
        {_card("Avg Fuel Flow", f'{cur_fuel:.4f}', 'kg/s', AMBER,
               f' &nbsp;{_badge(f"{fuel_change:+.1f}% vs base", fc_color)}')}
        {_card("Avg Torque", f'{cur_torque:,.0f}', 'kN m', CYAN,
               f' &nbsp;{_badge(f"{torque_change:+.1f}% vs base", tc_color)}')}
        {_card("Spec. Fuel Cons.", f'{sfc:.4f}', 'kg/s per MW', BLUE,
               f' &nbsp;{_badge(f"{eff_change:+.1f}%", ec_color)}')}
        {_card("Compressor Deg.", f'{comp_deg:.2f}%', 'from ideal', RED if comp_deg > 5 else AMBER)}
        {_card("Turbine Deg.", f'{turb_deg:.2f}%', 'from ideal', RED if turb_deg > 5 else AMBER)}
    </div>"""
    return html


# Chart builders
def chart_decay_trend(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["comp_decay"], name="Compressor Decay",
        line=dict(color=CYAN, width=2), fill="tozeroy",
        fillcolor="rgba(6,182,212,0.06)"))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["turb_decay"], name="Turbine Decay",
        line=dict(color=AMBER, width=2), fill="tozeroy",
        fillcolor="rgba(245,158,11,0.06)"))
    fig.add_hline(y=0.95, line_dash="dash", line_color=TEAL,
                  annotation_text="Healthy", annotation_font_color=TEAL,
                  annotation_font_size=10)
    fig.add_hline(y=0.90, line_dash="dash", line_color=RED,
                  annotation_text="Critical", annotation_font_color=RED,
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


def launch_digital_twin_dashboard(dt_instance=None, data_path=None):
    """
    Launch advanced Gradio dashboard with filtering and visualization.
    """
    # Load dataset
    if data_path is None:
        data_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "data.csv")
    
    RAW_DF = pd.read_csv(data_path)
    RAW_DF.columns = [c.strip() for c in RAW_DF.columns]
    
    # Column mapping
    COL_MAP = {
        "Lever position": "lever_pos",
        "Ship speed (v)": "ship_speed",
        "Gas Turbine (GT) shaft torque (GTT) [kN m]": "gt_torque",
        "GT rate of revolutions (GTn) [rpm]": "gt_rpm",
        "Gas Generator rate of revolutions (GGn) [rpm]": "gg_rpm",
        "Starboard Propeller Torque (Ts) [kN]": "ts",
        "Port Propeller Torque (Tp) [kN]": "tp",
        "Hight Pressure (HP) Turbine exit temperature (T48) [C]": "t48",
        "GT Compressor inlet air temperature (T1) [C]": "t1",
        "GT Compressor outlet air temperature (T2) [C]": "t2",
        "HP Turbine exit pressure (P48) [bar]": "p48",
        "GT Compressor inlet air pressure (P1) [bar]": "p1",
        "GT Compressor outlet air pressure (P2) [bar]": "p2",
        "GT exhaust gas pressure (Pexh) [bar]": "pexh",
        "Turbine Injecton Control (TIC) [%]": "tic",
        "Fuel flow (mf) [kg/s]": "fuel_flow",
        "GT Compressor decay state coefficient": "comp_decay",
        "GT Turbine decay state coefficient": "turb_decay",
    }
    RAW_DF.rename(columns=COL_MAP, inplace=True)
    
    BASELINES = RAW_DF.groupby("ship_speed").first().reset_index()
    SPEED_CHOICES = ["All Speeds"] + [f"{v} kn" for v in sorted(RAW_DF["ship_speed"].unique())]
    
    def filter_df(lever, speed_demand, tic_val):
        df = RAW_DF.copy()
        if lever is not None and lever != "All":
            df = df[df["lever_pos"] == float(lever)]
        if speed_demand is not None and speed_demand != "All Speeds":
            kn = int(speed_demand.replace(" kn", ""))
            df = df[df["ship_speed"] == kn]
        if tic_val is not None:
            low, high = tic_val
            df = df[(df["tic"] >= low) & (df["tic"] <= high)]
        if len(df) == 0:
            df = RAW_DF.copy()
        return df.reset_index(drop=True)
    
    def update_all(lever, speed, tic_range):
        df = filter_df(lever, speed, tic_range)
        return (
            sensor_html(df),
            health_html(df),
            efficiency_html(df, BASELINES),
            chart_decay_trend(df),
            chart_innovation(df),
            make_gauge(df["comp_decay"].mean(), "Compressor"),
            make_gauge(df["turb_decay"].mean(), "Turbine"),
            chart_fuel_efficiency(df),
            chart_temperatures(df),
            chart_pressures(df),
            chart_propulsion(df),
        )
    
    # CSS
    CSS = f"""
    .gradio-container {{
        background: {BG} !important;
        max-width: 1440px !important;
        color: {TEXT} !important;
    }}
    .gr-box, .gr-panel, .gr-form {{
        background: {BG} !important;
        border-color: {BORDER} !important;
    }}
    footer {{ display: none !important; }}
    .dark .tabitem {{ background: {BG} !important; }}
    .tab-nav button {{
        color: {TEXT} !important;
        background: {CARD} !important;
        border-color: {BORDER} !important;
        font-weight: 600 !important;
        font-size: 13px !important;
    }}
    .tab-nav button.selected {{
        color: {CYAN} !important;
        border-bottom: 2px solid {CYAN} !important;
        background: {CARD} !important;
    }}
    h1, h2, h3, h4, .prose h1, .prose h2, .prose h3 {{
        color: {TEXT} !important;
    }}
    .gr-button {{
        background: {CARD} !important;
        color: {TEXT} !important;
        border-color: {BORDER} !important;
    }}
    .gr-button:hover {{
        border-color: {CYAN} !important;
    }}
    .gr-input, .gr-dropdown, label, .label-wrap, span {{
        color: {TEXT} !important;
    }}
    .gr-input-label, .gr-slider label span {{
        color: {DIM} !important;
        font-size: 12px !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }}
    .section-label {{
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: {DIM};
        border-bottom: 1px solid {BORDER};
        padding-bottom: 6px;
        margin-bottom: 8px;
        font-weight: 600;
    }}
    """
    
    HEADER = f"""
    <div style="display:flex;align-items:center;gap:14px;padding:6px 0 14px 0;">
        <div>
            <svg width="38" height="38" viewBox="0 0 38 38" fill="none" xmlns="http://www.w3.org/2000/svg">
                <rect width="38" height="38" rx="8" fill="{CYAN}"/>
                <path d="M10 26 L19 10 L28 26 Z" fill="{BG}" stroke="{BG}" stroke-width="1.5" stroke-linejoin="round"/>
                <circle cx="19" cy="21" r="3" fill="{CYAN}"/>
            </svg>
        </div>
        <div>
            <h1 style="margin:0;font-size:22px;font-weight:700;color:{TEXT};font-family:Inter,system-ui,sans-serif;">
                Marine GT Propulsion Monitor
            </h1>
            <p style="margin:2px 0 0 0;font-size:12px;color:{DIM};">
                Condition-Based Maintenance Dashboard &mdash; {len(RAW_DF):,} records
            </p>
        </div>
        <div style="margin-left:auto;display:flex;align-items:center;gap:8px;">
            <span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:{TEAL};box-shadow:0 0 6px {TEAL};"></span>
            <span style="font-size:11px;color:{TEAL};font-weight:600;">SYSTEM ONLINE</span>
        </div>
    </div>
    """
    
    lever_choices = ["All"] + [str(v) for v in sorted(RAW_DF["lever_pos"].unique())]
    tic_min = float(RAW_DF["tic"].min())
    tic_max = float(RAW_DF["tic"].max())
    
    with gr.Blocks(title="Marine GT Propulsion Monitor") as demo:
        gr.HTML(HEADER)
        
        gr.HTML(f'<div class="section-label">Input Panel &mdash; Operator Controls</div>')
        with gr.Row():
            lever_dd = gr.Dropdown(choices=lever_choices, value="All",
                                   label="Lever Position (1-10)", interactive=True)
            speed_dd = gr.Dropdown(choices=SPEED_CHOICES, value="All Speeds",
                                   label="Ship Speed Demand", interactive=True)
            with gr.Column():
                gr.HTML(f'<div style="font-size:12px;color:{DIM};text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px;">Turbine Injection Control Range (%)</div>')
                with gr.Row():
                    tic_min_slider = gr.Slider(
                        minimum=tic_min,
                        maximum=tic_max,
                        value=tic_min,
                        label="Min TIC",
                        interactive=True,
                        step=0.1
                    )
                    tic_max_slider = gr.Slider(
                        minimum=tic_min,
                        maximum=tic_max,
                        value=tic_max,
                        label="Max TIC",
                        interactive=True,
                        step=0.1
                    )
            apply_btn = gr.Button("Apply Filters", variant="primary", scale=0)

        gr.HTML(f'<div class="section-label">Sensor Reading Display &mdash; 14 Measured Parameters</div>')
        sensor_out = gr.HTML()
        
        with gr.Tabs():
            with gr.Tab("Health Monitoring"):
                gr.HTML(f'<div class="section-label">Decay Coefficients &amp; Kalman Filter Metrics</div>')
                health_out = gr.HTML()
                
                with gr.Row():
                    comp_gauge = gr.Plot(label="Compressor Gauge")
                    turb_gauge = gr.Plot(label="Turbine Gauge")
                
                with gr.Row():
                    decay_plot = gr.Plot(label="Degradation Trend")
                    innov_plot = gr.Plot(label="Measurement Innovations")
            
            with gr.Tab("Operational Efficiency"):
                gr.HTML(f'<div class="section-label">Efficiency vs Baseline &amp; Fuel Consumption</div>')
                eff_out = gr.HTML()
                
                with gr.Row():
                    fuel_plot = gr.Plot(label="Fuel Consumption Trend")
            
            with gr.Tab("Sensor Trends"):
                with gr.Row():
                    temp_plot = gr.Plot(label="Temperature Profiles")
                    press_plot = gr.Plot(label="Pressure Profiles")
                with gr.Row():
                    prop_plot = gr.Plot(label="Propeller Torque")
        
        outputs = [
            sensor_out, health_out, eff_out,
            decay_plot, innov_plot, comp_gauge, turb_gauge,
            fuel_plot, temp_plot, press_plot, prop_plot,
        ]

        def update_with_range(lever, speed, tic_min_val, tic_max_val):
            return update_all(lever, speed, (tic_min_val, tic_max_val))

        apply_btn.click(fn=update_with_range, inputs=[lever_dd, speed_dd, tic_min_slider, tic_max_slider], outputs=outputs)
        lever_dd.change(fn=update_with_range, inputs=[lever_dd, speed_dd, tic_min_slider, tic_max_slider], outputs=outputs)
        speed_dd.change(fn=update_with_range, inputs=[lever_dd, speed_dd, tic_min_slider, tic_max_slider], outputs=outputs)
        tic_min_slider.change(fn=update_with_range, inputs=[lever_dd, speed_dd, tic_min_slider, tic_max_slider], outputs=outputs)
        tic_max_slider.change(fn=update_with_range, inputs=[lever_dd, speed_dd, tic_min_slider, tic_max_slider], outputs=outputs)
        demo.load(fn=update_with_range, inputs=[lever_dd, speed_dd, tic_min_slider, tic_max_slider], outputs=outputs)
    
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        css=CSS,
        theme=gr.themes.Base()
    )