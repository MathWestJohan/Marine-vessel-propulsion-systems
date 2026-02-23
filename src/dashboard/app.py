import os
import pandas as pd
import gradio as gr
import warnings

# Suppress sklearn feature name warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

from dashboard.components import (
    DIM, CYAN, sensor_html, health_html, efficiency_html,
    get_css, header_html,
)
from dashboard.charts import (
    chart_decay_trend, chart_innovation, chart_fuel_efficiency,
    chart_temperatures, chart_pressures, chart_propulsion, make_gauge,
)
from dashboard.chatbot import respond, get_system_context

# Column mapping for raw CSV
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

def _load_data(data_path=None):
    if data_path is None:
        data_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "data.csv")
    df = pd.read_csv(data_path)
    # Reverse to match chronological order (Healthy -> Degraded)
    df = df.iloc[::-1].reset_index(drop=True)
    df.columns = [c.strip() for c in df.columns]
    df.rename(columns=COL_MAP, inplace=True)
    return df


def _add_predictions(df, dt_instance):
    """
    Run ML models on sensor features and add predicted decay columns.
    Keeps the original CSV columns as ground truth for comparison.
    """
    if dt_instance is None:
        return df
    
    df = df.copy()
    
    model = dt_instance.compressor_model
    if hasattr(model, "n_features_in_"):
        n_expected = model.n_features_in_
    else:
        n_expected = len(FEATURE_COLS)
    
    feature_cols = [c for c in FEATURE_COLS if c in df.columns][:n_expected]
    
    if len(feature_cols) < n_expected:
        print(f" Feature mismatch: Model expects {n_expected}, found {len(feature_cols)}")
        print(f" Available: {feature_cols}")
        return df
    
    # Extract sensor features for ML prediction
    X_raw = df[feature_cols]
    
    # Scale for compressor model (only if SVM won)
    if dt_instance.comp_scaler is not None:
        # Using .values to avoid "feature names mismatch" error with scikit-learn
        X_comp = dt_instance.comp_scaler.transform(X_raw.values)
    else:
        X_comp = X_raw.values

    # Scale for turbine model (only if SVM won)
    if dt_instance.turb_scaler is not None:
        # Using .values to avoid "feature names mismatch" error with scikit-learn
        X_turb = dt_instance.turb_scaler.transform(X_raw.values)
    else:
        X_turb = X_raw.values

    # Run ML model predictions
    # SVM models (from sklearn) work fine with numpy arrays
    df["comp_decay_pred"] = dt_instance.compressor_model.predict(X_comp)
    df["turb_decay_pred"] = dt_instance.turbine_model.predict(X_turb)

    # Store ground truth for comparison
    df["comp_decay_actual"] = df["comp_decay"]
    df["turb_decay_actual"] = df["turb_decay"]

    # Replace with predictions so all downstream code uses them
    df["comp_decay"] = df["comp_decay_pred"]
    df["turb_decay"] = df["turb_decay_pred"]
    
    return df

def generate_report(df, dt_instance):
    if df is None:
        return None
    context = get_system_context(df, dt_instance)
    report_path = "maintenance_report.txt"
    with open(report_path, "w") as f:
        f.write("=== MARINE PROPULSION MAINTENANCE REPORT ===\n")
        f.write(context)
        f.write("\nRECOMMENDATION:\n")
        status = {
            "comp_alert": df["comp_decay"].mean() < 0.975,
            "turb_alert": df["turb_decay"].mean() < 0.975
        }
        f.write(dt_instance.get_maintenance_recommendation(status))
    return report_path

def launch_dashboard(dt_instance=None, data_path=None):
    """Launch the Gradio dashboard."""
    RAW_DF = _load_data(data_path)
    BASELINES = RAW_DF.groupby("ship_speed").first().reset_index()
    speed_choices = ["All"] + [f"{v} kn" for v in sorted(RAW_DF["ship_speed"].unique())]
    tic_min = float(RAW_DF["tic"].min())
    tic_max = float(RAW_DF["tic"].max())
    
    # Check if models are available
    has_models = dt_instance is not None and hasattr(dt_instance, 'compressor_model')
    if has_models:
        print(" ML models detected — dashboard will show model predictions")
        n = dt_instance.compressor_model.n_features_in_ if hasattr(dt_instance.compressor_model, 'n_features_in_') else '?'
        print(f" Model expects {n} features, FEATURE_COLS has {len(FEATURE_COLS)}")
    else:
        print(" No ML models — dashboard will show raw CSV values")

    def filter_df(speed, tic_val):
        df = RAW_DF.copy()
        if speed is not None and speed != "All":
            df = df[df["ship_speed"] == float(speed.replace(" kn", ""))]
        if tic_val is not None:
            low, high = tic_val
            df = df[(df["tic"] >= low) & (df["tic"] <= high)]
        if len(df) == 0:
            df = RAW_DF.copy()
        return df.reset_index(drop=True)

    def update_all(speed, tic_min_val, tic_max_val, fault_offsets=None):
        df = filter_df(speed, (tic_min_val, tic_max_val))
        
        # Apply simulated faults if any
        if fault_offsets:
            for sensor, offset in fault_offsets.items():
                if sensor in df.columns:
                    df[sensor] = df[sensor] * (1 + offset/100)

        # Run ML predictions on filtered data
        df = _add_predictions(df, dt_instance)
        
        # Detect maintenance jumps for the current view
        if has_models and dt_instance:
            dt_instance.maintenance_history = []
            dt_instance.health_history = []
            dt_instance.last_health = {"compressor": 1.0, "turbine": 1.0}
            
            # Vectorized history for status cards
            comp_vals = df["comp_decay"].values
            turb_vals = df["turb_decay"].values
            
            # Detect jumps efficiently
            for i in range(1, len(df)):
                if comp_vals[i] > comp_vals[i-1] + 0.01 or turb_vals[i] > turb_vals[i-1] + 0.01:
                    dt_instance._detect_jumps(comp_vals[i], turb_vals[i])

        source = "ML Model Predictions" if has_models else "Raw CSV"
        maint_hist = getattr(dt_instance, "maintenance_history", [])
        
        print(f"\n--- Updating dashboard with {len(df)} records (Source: {source}) ---")
        print(f" Speed filter: {speed}, TIC range: [{tic_min_val}, {tic_max_val}]")
        print(f" Available features: {', '.join([c for c in FEATURE_COLS if c in df.columns])}")
        return (
            sensor_html(df),
            health_html(df, dt_instance, source_label=source),
            efficiency_html(df, BASELINES),
            chart_decay_trend(df, maintenance_history=maint_hist),
            chart_innovation(df),
            make_gauge(df["comp_decay"].mean(), "Compressor"),
            make_gauge(df["turb_decay"].mean(), "Turbine"),
            chart_fuel_efficiency(df),
            chart_temperatures(df),
            chart_pressures(df),
            chart_propulsion(df),
            df,
        )

    with gr.Blocks(title="Marine GT Propulsion Monitor") as demo:
        current_df = gr.State()
        fault_offsets = gr.State({})
        gr.HTML(header_html(len(RAW_DF)))

        with gr.Row():
            with gr.Column(scale=3):
                gr.HTML(f'<div class="section-label">Input Panel &mdash; Operating Conditions</div>')
                with gr.Row():
                    speed_dd = gr.Dropdown(
                        choices=speed_choices, value="All",
                        label="Ship Speed (knots)", interactive=True,
                    )
                    with gr.Column():
                        gr.HTML(f'<div style="font-size:12px;color:{DIM};text-transform:uppercase;margin-bottom:4px;">TIC Range (%)</div>')
                        with gr.Row():
                            tic_min_slider = gr.Slider(minimum=tic_min, maximum=tic_max, value=tic_min, label="Min", interactive=True)
                            tic_max_slider = gr.Slider(minimum=tic_min, maximum=tic_max, value=tic_max, label="Max", interactive=True)
                apply_btn = gr.Button("Apply Filters & Refresh Predictions", variant="primary")
            
            with gr.Column(scale=2):
                gr.HTML(f'<div class="section-label">Fault Simulation & Reporting</div>')
                with gr.Row():
                    fault_type = gr.Dropdown(
                        choices=["None", "T48 Sensor Spike (+10%)", "P2 Pressure Drop (-10%)", "Fuel Leak (+5%)"],
                        value="None", label="Inject Fault"
                    )
                    report_btn = gr.Button("Generate Report")
                report_file = gr.File(label="Download Report", interactive=False)

        gr.HTML(f'<div class="section-label">Sensor Reading Display &mdash; 14 Measured Parameters</div>')
        sensor_out = gr.HTML()

        with gr.Tabs():
            with gr.Tab("Health Monitoring"):
                gr.HTML(f'<div class="section-label">Decay Coefficients &amp; Health Metrics</div>')
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

            with gr.Tab("AI Assistant"):
                gr.HTML(f'<div class="section-label">Digital Twin AI Assistant</div>')
                chatbot = gr.Chatbot(label="Marine Advisor", height=500)
                with gr.Row():
                    msg = gr.Textbox(
                        label="Ask about system health or maintenance...",
                        placeholder="e.g., What is the current state of the compressor?",
                        scale=4
                    )
                    clear = gr.Button("Clear Chat", scale=1)

        outputs = [
            sensor_out, health_out, eff_out,
            decay_plot, innov_plot, comp_gauge, turb_gauge,
            fuel_plot, temp_plot, press_plot, prop_plot,
            current_df,
        ]
        inputs = [speed_dd, tic_min_slider, tic_max_slider, fault_offsets]

        def handle_fault(choice):
            offsets = {}
            if "T48" in choice: offsets["t48"] = 10
            elif "P2" in choice: offsets["p2"] = -10
            elif "Fuel" in choice: offsets["fuel_flow"] = 5
            return offsets

        def chat_wrapper(message, history, df):
            if history is None:
                history = []
            return respond(message, history, df, dt_twin=dt_instance)

        fault_type.change(handle_fault, fault_type, fault_offsets).then(
            fn=update_all, inputs=inputs, outputs=outputs
        )
        
        report_btn.click(lambda df: generate_report(df, dt_instance), current_df, report_file)

        msg.submit(chat_wrapper, [msg, chatbot, current_df], [chatbot])
        msg.submit(lambda: "", None, [msg])
        clear.click(lambda: [], None, chatbot, queue=False)

        apply_btn.click(fn=update_all, inputs=inputs, outputs=outputs)
        speed_dd.change(fn=update_all, inputs=inputs, outputs=outputs)
        tic_min_slider.change(fn=update_all, inputs=inputs, outputs=outputs)
        tic_max_slider.change(fn=update_all, inputs=inputs, outputs=outputs)
        demo.load(fn=update_all, inputs=inputs, outputs=outputs)

        print("\n--- Launching Digital Twin Dashboard ---")
        print(" Note: Dashboard analyses historical CSV data (not live telemetry)")
        demo.launch(server_name="127.0.0.1", server_port=7860, css=get_css())
    