import numpy as np
from models.DigitalTwin import detect_faults, estimate_remaining_life

# ── Theme tokens ──
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
    """Display 14 measured parameters."""
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
    html += "</div>"
    return html


def health_html(df):
    """Health status cards with Kalman metrics."""
    comp_val = df["comp_decay"].mean()
    turb_val = df["turb_decay"].mean()
    cc, cl = _status_color(comp_val)
    tc, tl = _status_color(turb_val)

    faults, metrics = detect_faults(df)
    comp_f, turb_f = faults[0], faults[1]

    comp_trend = np.polyfit(range(len(df)), df["comp_decay"].values, 1)
    turb_trend = np.polyfit(range(len(df)), df["turb_decay"].values, 1)
    comp_maint = estimate_remaining_life(comp_trend[0], comp_val)
    turb_maint = estimate_remaining_life(turb_trend[0], turb_val)

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
    """Operational efficiency HTML."""
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
        fuel_change = torque_change = eff_change = 0
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


def get_css():
    """Return the full CSS string for the Gradio app."""
    return f"""
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


def header_html(record_count):
    """Return the header banner HTML."""
    return f"""
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
                Condition-Based Maintenance Dashboard &mdash; {record_count:,} records
            </p>
        </div>
        <div style="margin-left:auto;display:flex;align-items:center;gap:8px;">
            <span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:{TEAL};box-shadow:0 0 6px {TEAL};"></span>
            <span style="font-size:11px;color:{TEAL};font-weight:600;">SYSTEM ONLINE</span>
        </div>
    </div>
    """