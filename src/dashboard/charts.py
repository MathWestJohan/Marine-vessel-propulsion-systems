import plotly.graph_objects as go
import numpy as np

# Use theme tokens from components
from .components import BG_COLOR, CARD_COLOR, BORDER_COLOR, TEXT_PRIMARY, TEXT_DIM, ACCENT_CYAN, ACCENT_AMBER, ACCENT_TEAL

def create_main_trend_chart(df, turbine_events, comp_threshold=0.95, turb_threshold=0.975):
    """
    Creates a shared trend chart showing Turbine Sawtooth and Compressor Long Decline.
    """
    fig = go.Figure()

    # Compressor Line (Long Decline)
    fig.add_trace(go.Scatter(
        x=df.index, y=df['comp_decay'],
        name="Compressor (Single Cycle)",
        line=dict(color=ACCENT_CYAN, width=3),
        hovertemplate="Sample: %{x}<br>Health: %{y:.4f}"
    ))

    # Turbine Line (Sawtooth)
    fig.add_trace(go.Scatter(
        x=df.index, y=df['turb_decay'],
        name="Gas Turbine (Sawtooth)",
        line=dict(color=ACCENT_AMBER, width=2),
        hovertemplate="Sample: %{x}<br>Health: %{y:.4f}"
    ))

    # Add Maintenance Thresholds
    fig.add_hline(y=turb_threshold, line_dash="dash", line_color=ACCENT_AMBER, 
                  annotation_text="GT Maint. Threshold", annotation_position="top left",
                  annotation_font_color=ACCENT_AMBER)
    
    fig.add_hline(y=comp_threshold, line_dash="dash", line_color=ACCENT_CYAN,
                  annotation_text="Comp Maint. Threshold", annotation_position="bottom left",
                  annotation_font_color=ACCENT_CYAN)

    # Mark Turbine Maintenance Events (The "Sawtooth" jumps)
    for event_idx in turbine_events:
        fig.add_vline(x=event_idx, line_width=1, line_dash="dot", line_color=ACCENT_TEAL)
        fig.add_annotation(x=event_idx, y=1.01, text="GT MAINT", showarrow=False, 
                         font=dict(color=ACCENT_TEAL, size=10), textangle=-90)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=CARD_COLOR,
        font=dict(color=TEXT_PRIMARY),
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(gridcolor=BORDER_COLOR, title="Operational Time (Samples)"),
        yaxis=dict(gridcolor=BORDER_COLOR, title="Health Coefficient", range=[0.94, 1.02]),
        hovermode="x unified"
    )
    return fig

def create_gauge(value, label, color, threshold):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': label, 'font': {'size': 16, 'color': TEXT_PRIMARY}},
        number={'font': {'color': color, 'family': "JetBrains Mono"}, 'valueformat': ".4f"},
        gauge={
            'axis': {'range': [0.94, 1.0], 'tickwidth': 1, 'tickcolor': TEXT_DIM},
            'bar': {'color': color},
            'bgcolor': CARD_COLOR,
            'borderwidth': 2,
            'bordercolor': BORDER_COLOR,
            'steps': [
                {'range': [0.94, threshold], 'color': 'rgba(239, 68, 68, 0.1)'},
                {'range': [threshold, 1.0], 'color': 'rgba(45, 212, 191, 0.1)'}
            ],
            'threshold': {
                'line': {'color': TEXT_PRIMARY, 'width': 4},
                'thickness': 0.75,
                'value': threshold
            }
        }
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=30, r=30, t=50, b=20),
        height=220
    )
    return fig

def create_sensor_impact_chart(df_selected, df_baseline):
    """
    Compares selected cycle sensors against Cycle 1 baseline at each speed step.
    """
    # Key sensors to track for impact
    impact_sensors = ["t48", "t2", "p48", "p2", "fuel_flow", "gt_torque"]
    sensor_names = {
        "t48": "Turbine Exit Temp",
        "t2": "Compressor Outlet Temp",
        "p48": "Turbine Exit Press",
        "p2": "Compressor Outlet Press",
        "fuel_flow": "Fuel Flow",
        "gt_torque": "Turbine Torque"
    }
    
    # Calculate average deviation across the speed sweep (27 -> 3)
    deviations = []
    labels = []
    
    for s in impact_sensors:
        # Match by ship_speed to ensure apples-to-apples comparison
        # We group by speed and take the mean to handle any variations
        base_means = df_baseline.groupby("ship_speed")[s].mean()
        curr_means = df_selected.groupby("ship_speed")[s].mean()
        
        # Calculate % change
        pct_change = ((curr_means - base_means) / base_means).mean() * 100
        
        if np.isfinite(pct_change):
            deviations.append(pct_change)
            labels.append(sensor_names[s])

    fig = go.Figure(go.Bar(
        x=deviations,
        y=labels,
        orientation='h',
        marker=dict(
            color=[ACCENT_AMBER if d > 0 else ACCENT_CYAN for d in deviations],
            line=dict(color=TEXT_PRIMARY, width=1)
        ),
        hovertemplate="Deviation: %{x:.2f}%<extra></extra>"
    ))

    fig.update_layout(
        title="Average Sensor Deviation vs. Baseline (Cycle 1)",
        template="plotly_dark",
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=CARD_COLOR,
        font=dict(color=TEXT_PRIMARY),
        xaxis=dict(title="Percentage Change (%)", gridcolor=BORDER_COLOR, zerolinecolor=TEXT_PRIMARY),
        yaxis=dict(gridcolor=BORDER_COLOR),
        margin=dict(l=150, r=40, t=60, b=40),
        height=400
    )
    
    return fig
