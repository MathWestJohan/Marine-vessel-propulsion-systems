import numpy as np

#  Theme Modern Dark Marine
BG_COLOR = "#0f172a"
CARD_COLOR = "#1e293b"
BORDER_COLOR = "#334155"
TEXT_PRIMARY = "#f8fafc"
TEXT_DIM = "#94a3b8"
ACCENT_CYAN = "#06b6d4"   # Compressor
ACCENT_AMBER = "#f59e0b"  # Turbine
ACCENT_TEAL = "#2dd4bf"   # Healthy
ACCENT_RED = "#ef4444"    # Critical

def get_css():
    return f"""
    .gradio-container {{ background-color: {BG_COLOR} !important; color: {TEXT_PRIMARY} !important; }}
    .status-card {{
        background: {CARD_COLOR};
        border: 1px solid {BORDER_COLOR};
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 12px;
    }}
    .metric-value {{ font-family: 'JetBrains Mono', monospace; font-size: 2.5rem; font-weight: 700; }}
    .metric-label {{ text-transform: uppercase; font-size: 0.75rem; letter-spacing: 0.1em; color: {TEXT_DIM}; }}
    .cycle-badge {{
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.7rem;
        font-weight: 700;
        background: rgba(45, 212, 191, 0.1);
        color: {ACCENT_TEAL};
        border: 1px solid {ACCENT_TEAL};
    }}
    """

def health_card(name, value, cycle_info, color, status="HEALTHY"):
    status_color = ACCENT_TEAL if status == "HEALTHY" else ACCENT_RED
    return f"""
    <div class="status-card">
        <div style="display: flex; justify-content: space-between; align-items: start;">
            <div>
                <div class="metric-label">{name} Health Index</div>
                <div class="metric-value" style="color: {color}">{value:.4f}</div>
            </div>
            <div style="text-align: right;">
                <div class="cycle-badge" style="color: {status_color}; border-color: {status_color}">{status}</div>
                <div style="font-size: 0.75rem; color: {TEXT_DIM}; margin-top: 8px;">{cycle_info}</div>
            </div>
        </div>
        <div style="margin-top: 15px; background: {BORDER_COLOR}; height: 4px; border-radius: 2px;">
            <div style="background: {color}; width: {value*100}%; height: 100%; border-radius: 2px;"></div>
        </div>
    </div>
    """

def sensor_grid(metrics):
    html = '<div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 12px;">'
    for label, val, unit in metrics:
        html += f"""
        <div style="background: {CARD_COLOR}; padding: 12px; border-radius: 8px; border: 1px solid {BORDER_COLOR};">
            <div class="metric-label" style="font-size: 0.6rem;">{label}</div>
            <div style="font-size: 1.1rem; font-weight: 600; color: {TEXT_PRIMARY};">{val} <span style="font-size: 0.7rem; color: {TEXT_DIM};">{unit}</span></div>
        </div>
        """
    html += "</div>"
    return html
