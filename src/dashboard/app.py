import os
import pandas as pd
import numpy as np
import gradio as gr
import warnings

# Suppress sklearn feature name warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

from .components import (
    get_css, health_card, sensor_grid, 
    ACCENT_CYAN, ACCENT_AMBER, ACCENT_TEAL, ACCENT_RED
)
from .charts import create_main_trend_chart, create_gauge, create_sensor_impact_chart
from .chatbot import chat_stream

# --- Constants & Config ---
# We use a mapping that handles the hidden non-breaking spaces found in the raw CSV
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

FEATURE_COLS = [
    "lever_pos", "ship_speed", "gt_torque", "gt_rpm", "gg_rpm",
    "ts", "tp", "t48", "t2", "p48", "p2", "pexh",
    "tic", "fuel_flow",
]

def load_data():
    data_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "data.csv")
    df = pd.read_csv(data_path)
    
    # Robust cleaning: strip standard whitespace AND non-breaking spaces (\xa0)
    df.columns = [c.strip().replace('\xa0', '') for c in df.columns]
    
    # Rename to our internal standardized names
    df.rename(columns=COL_MAP, inplace=True)
    
    # Reverse to match chronological order (Healthy -> Degraded)
    df = df.iloc[::-1].reset_index(drop=True)
    
    # Identify Mission Cycles (each 27 -> 3 knot sweep)
    cycles = []
    current_cycle = 0
    for i in range(len(df)):
        if df['ship_speed'].iloc[i] == 27:
            current_cycle += 1
        cycles.append(current_cycle)
    df['mission_cycle'] = cycles
    return df

def calculate_rul(df, current_val, target_col='comp_decay', threshold=0.95):
    """
    Predicts Remaining Useful Life in terms of Mission Cycles (27-3 kn sweeps).
    """
    if target_col == 'turb_decay':
        total_drop_range = 1.0 - 0.975 
        remaining_health_budget = current_val - threshold
        if remaining_health_budget <= 0: return "Maintenance Due"
        mission_cycles_per_maint = 26.0
        cycles_left = (remaining_health_budget / total_drop_range) * mission_cycles_per_maint
        return f"{max(0, cycles_left):.1f} Cycles"
    else:
        y = df[target_col].values
        slope, _ = np.polyfit(np.arange(len(y)), y, 1)
        if slope >= 0: return "Stable"
        samples_left = (threshold - current_val) / slope
        cycles_left = samples_left / 9.0
        return f"{max(0, cycles_left):.1f} Cycles"

def detect_turbine_cycles(df):
    turbine_events = []
    jump_threshold = 0.01
    for i in range(1, len(df)):
        if df['turb_decay'].iloc[i] > df['turb_decay'].iloc[i-1] + jump_threshold:
            turbine_events.append(i)
    return turbine_events

def launch_dashboard(dt_instance):
    RAW_DF = load_data()
    MAX_CYCLES = RAW_DF['mission_cycle'].max()
    
    def update_dashboard(cycle_selection, downsample, random_trigger, active_faults):
        # 1. Filter Data by Mission Cycle
        df_baseline = RAW_DF[RAW_DF['mission_cycle'] == 1]
        
        if cycle_selection == "All Mission Data":
            df_view = RAW_DF
            snapshot_label = "Full Life-Cycle Profile"
            df_selected_for_impact = RAW_DF[RAW_DF['mission_cycle'] == MAX_CYCLES]
        else:
            try:
                # Correctly parse "Mission Cycle 10" -> 10
                cycle_num = int(cycle_selection.split(" ")[-1])
            except (IndexError, ValueError):
                cycle_num = 1
            df_view = RAW_DF[RAW_DF['mission_cycle'] == cycle_num]
            snapshot_label = f"Mission Cycle {cycle_num} Detail (27 \u2192 3 kn)"
            df_selected_for_impact = df_view

        # 2. Select Snapshot
        if len(df_view) > 0:
            random_idx = np.random.randint(0, len(df_view))
            current_snapshot = df_view.iloc[random_idx].copy()
            snapshot_label += f" | {current_snapshot['ship_speed']:.0f} kn"
        else:
            return [None] * 13

        # --- Apply Injected Faults ---
        fault_info = ""
        if active_faults:
            for sensor, offset_pct in active_faults.items():
                if sensor in current_snapshot:
                    current_snapshot[sensor] = current_snapshot[sensor] * (1 + offset_pct / 100)
                    fault_info += f"FAULT: {sensor} shifted {offset_pct:+.1f}% | "

        # 3. ML Model Predictions (Virtual Sensing)
        feature_data = current_snapshot[FEATURE_COLS]
        
        if dt_instance:
            pred = dt_instance.predict_health(feature_data)
            comp_health = pred["compressor_health"]
            turb_health = pred["turbine_health"]
            source_tag = "ML Prediction"
        else:
            comp_health = float(current_snapshot["comp_decay"])
            turb_health = float(current_snapshot["turb_decay"])
            source_tag = "Ground Truth"
        
        # Calculate RUL
        comp_rul = calculate_rul(RAW_DF, comp_health, target_col='comp_decay', threshold=0.95)
        turb_rul = calculate_rul(RAW_DF, turb_health, target_col='turb_decay', threshold=0.975)
        
        turbine_events = detect_turbine_cycles(RAW_DF)
        num_gt_maint = sum(1 for e in turbine_events if e <= current_snapshot.name)
        
        comp_status = "HEALTHY" if comp_health >= 0.95 else "CRITICAL"
        turb_status = "HEALTHY" if turb_health >= 0.975 else "CRITICAL"

        # 4. HTML Components
        comp_card = health_card(f"Compressor ({source_tag})", comp_health, f"Maint. in: {comp_rul}", ACCENT_CYAN, comp_status)
        turb_card = health_card(f"Gas Turbine ({source_tag})", turb_health, f"Maint. in: {turb_rul}", ACCENT_AMBER, turb_status)
        
        sensors = [
            ("Sample Index", f"{current_snapshot.name}", "#"),
            ("Current Speed", f"{current_snapshot['ship_speed']:.0f}", "kn"),
            ("GT Torque", f"{current_snapshot['gt_torque']:,.0f}", "kN m"),
            ("Fuel Flow", f"{current_snapshot['fuel_flow']:.4f}", "kg/s"),
            ("Exit Temp", f"{current_snapshot['t48']:.1f}", "\u00b0C"),
            ("Outlet Pres", f"{current_snapshot['p2']:.3f}", "bar"),
            ("GT RPM", f"{current_snapshot['gt_rpm']:,.0f}", "rpm"),
            ("GG RPM", f"{current_snapshot['gg_rpm']:,.0f}", "rpm"),
            ("Inlet Temp", f"{current_snapshot['t1']:.1f}", "\u00b0C"),
            ("Outlet Temp", f"{current_snapshot['t2']:.1f}", "\u00b0C"),
            ("Inlet Pres", f"{current_snapshot['p1']:.3f}", "bar"),
            ("Exh Pres", f"{current_snapshot['pexh']:.3f}", "bar"),
        ]
        
        header_color = ACCENT_RED if fault_info else ACCENT_TEAL
        grid_html = f"<div style='margin-bottom:10px; color:{header_color}; font-weight:bold;'>{snapshot_label} | Sample #{current_snapshot.name}</div>"
        if fault_info:
            grid_html += f"<div style='background:rgba(239,68,68,0.1); border:1px solid {ACCENT_RED}; padding:5px; margin-bottom:10px; font-size:0.8rem; color:{ACCENT_RED};'>{fault_info}</div>"
        grid_html += sensor_grid(sensors)

        # 5. Plots
        df_plot = df_view
        if downsample:
            step = max(1, len(df_plot) // 1000)
            df_plot = df_plot.iloc[::step]
            
        trend_fig = create_main_trend_chart(df_plot if cycle_selection != "All Mission Data" else RAW_DF, turbine_events)
        comp_gauge = create_gauge(comp_health, "Compressor Health Index", ACCENT_CYAN, 0.95)
        turb_gauge = create_gauge(turb_health, "GT Health Index", ACCENT_AMBER, 0.975)
        impact_fig = create_sensor_impact_chart(df_selected_for_impact, df_baseline)

        return (
            comp_card, turb_card, grid_html, 
            trend_fig, comp_gauge, turb_gauge, impact_fig,
            active_faults, current_snapshot, comp_health, turb_health, comp_rul, turb_rul
        )

    with gr.Blocks(title="Propulsion Digital Twin") as demo:
        gr.Markdown(f"# Marine Propulsion Digital Twin <span style='color:{ACCENT_TEAL}; font-size: 0.8rem; margin-left: 10px;'>ONLINE</span>")
        
        random_trigger = gr.State(0)
        active_faults = gr.State({})
        
        # States for Chatbot context
        chat_snapshot = gr.State()
        chat_comp_health = gr.State()
        chat_turb_health = gr.State()
        chat_comp_rul = gr.State()
        chat_turb_rul = gr.State()

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Mission Context")
                cycle_dd = gr.Dropdown(
                    choices=["All Mission Data"] + [f"Mission Cycle {i}" for i in range(1, MAX_CYCLES + 1)],
                    value="All Mission Data", label="Mission Cycle Selection"
                )
                refresh_btn = gr.Button("Random Snapshot in Cycle", variant="secondary")
                downsample_chk = gr.Checkbox(value=True, label="Enable Downsampling")
                gr.Markdown("---")
                gr.Markdown("### Technical Insights")
                gr.Markdown(f"""
                - **Mission Cycle:** One cycle = A full speed sweep from 27 knots to 3 knots.
                - **Maintenance Tracking:** GT maintenance resets are tracked via cumulative count.
                - **Fault Injection:** Use the Lab tab to test model robustness.
                - **AI Agent:** Chat with the Chief Engineer in the new tab.
                """)

            with gr.Column(scale=3):
                with gr.Tabs():
                    with gr.Tab("System Health Snapshot"):
                        with gr.Row():
                            comp_out = gr.HTML()
                            turb_out = gr.HTML()
                        sensor_out = gr.HTML()
                        with gr.Row():
                            comp_gauge_out = gr.Plot()
                            turb_gauge_out = gr.Plot()

                    with gr.Tab("Historical Mission Trends"):
                        main_chart = gr.Plot()

                    with gr.Tab("Sensor Impact Analysis"):
                        gr.Markdown("### Deviation vs. Healthy Baseline (Cycle 1)")
                        impact_chart = gr.Plot()
                    
                    with gr.Tab("Fault Simulation Lab"):
                        gr.Markdown("### Manual Fault Injection")
                        with gr.Row():
                            fault_sensor = gr.Dropdown(
                                choices=[("Turbine Exit Temp (T48)", "t48"), 
                                         ("Compressor Outlet Pres (P2)", "p2"), 
                                         ("Fuel Flow (mf)", "fuel_flow")],
                                label="Select Sensor", value="t48"
                            )
                            fault_magnitude = gr.Slider(minimum=-20, maximum=20, value=0, step=1, label="Offset (%)")
                        
                        with gr.Row():
                            apply_fault_btn = gr.Button("Apply Fault Offset", variant="primary")
                            clear_fault_btn = gr.Button("Clear All Faults", variant="secondary")
                        
                        gr.Markdown("**Tip:** Apply a fault, then check the 'Snapshot' tab to see how the health index and countdown drop.")
                        
                    with gr.Tab("AI Chief Engineer"):
                        gr.Markdown("### Ask the AI Chief Engineer")
                        gr.Markdown("The AI has real-time access to the current snapshot, maintenance schedule, and injected faults.")
                        gr.ChatInterface(
                            fn=chat_stream,
                            additional_inputs=[chat_snapshot, chat_comp_health, chat_turb_health, chat_comp_rul, chat_turb_rul, active_faults]
                        )

        # Wire up events
        inputs = [cycle_dd, downsample_chk, random_trigger, active_faults]
        outputs = [
            comp_out, turb_out, sensor_out, main_chart, comp_gauge_out, turb_gauge_out, impact_chart, active_faults,
            chat_snapshot, chat_comp_health, chat_turb_health, chat_comp_rul, chat_turb_rul
        ]
        
        def trigger_refresh(current_val): return current_val + 1
        
        def handle_apply_fault(sensor, magnitude, current_faults):
            new_faults = dict(current_faults)
            if magnitude == 0:
                if sensor in new_faults: del new_faults[sensor]
            else:
                new_faults[sensor] = magnitude
            return new_faults

        def handle_clear_faults(): return {}

        apply_fault_btn.click(handle_apply_fault, [fault_sensor, fault_magnitude, active_faults], active_faults).then(
            update_dashboard, inputs, outputs
        )
        
        clear_fault_btn.click(handle_clear_faults, None, active_faults).then(
            update_dashboard, inputs, outputs
        )

        refresh_btn.click(trigger_refresh, random_trigger, random_trigger)
        cycle_dd.change(update_dashboard, inputs, outputs)
        random_trigger.change(update_dashboard, inputs, outputs)
        demo.load(update_dashboard, inputs, outputs)

    demo.launch(server_name="127.0.0.1", server_port=7860, css=get_css())
