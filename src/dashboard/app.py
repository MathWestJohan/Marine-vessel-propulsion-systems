import os
import pandas as pd
import gradio as gr

from dashboard.components import (
    DIM, CYAN, sensor_html, health_html, efficiency_html,
    get_css, header_html,
)
from dashboard.charts import (
    chart_decay_trend, chart_innovation, chart_fuel_efficiency,
    chart_temperatures, chart_pressures, chart_propulsion, make_gauge,
)

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
    X = df[feature_cols].values
    
    # Scale for compressor model (only if SVM won)
    if dt_instance.comp_scaler is not None:
        X_comp = dt_instance.comp_scaler.transform(X)
    else:
        X_comp = X

    # Scale for turbine model (only if SVM won)
    if dt_instance.turb_scaler is not None:
        X_turb = dt_instance.turb_scaler.transform(X)
    else:
        X_turb = X

    # Run ML model predictions
    df["comp_decay_pred"] = dt_instance.compressor_model.predict(X_comp)
    df["turb_decay_pred"] = dt_instance.turbine_model.predict(X_turb)

    # Store ground truth for comparison
    df["comp_decay_actual"] = df["comp_decay"]
    df["turb_decay_actual"] = df["turb_decay"]

    # Replace with predictions so all downstream code uses them
    df["comp_decay"] = df["comp_decay_pred"]
    df["turb_decay"] = df["turb_decay_pred"]
    
    return df

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

    def update_all(speed, tic_min_val, tic_max_val):
        df = filter_df(speed, (tic_min_val, tic_max_val))
        
        # Run ML predictions on filtered data
        df = _add_predictions(df, dt_instance)
        
        source = "ML Model Predictions" if has_models else "Raw CSV"
        
        print(f"\n--- Updating dashboard with {len(df)} records (Source: {source}) ---")
        print(f" Speed filter: {speed}, TIC range: [{tic_min_val}, {tic_max_val}]")
        print(f" Available features: {', '.join([c for c in FEATURE_COLS if c in df.columns])}")
        return (
            sensor_html(df),
            health_html(df, dt_instance, source_label=source),
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

    with gr.Blocks(css=get_css(), title="Marine GT Propulsion Monitor") as demo:
        gr.HTML(header_html(len(RAW_DF)))

        gr.HTML(f'<div class="section-label">Input Panel &mdash; Operating Conditions</div>')
        with gr.Row():
            speed_dd = gr.Dropdown(
                choices=speed_choices,
                value="All",
                label="Ship Speed (knots)",
                interactive=True,
            )
            with gr.Column():
                gr.HTML(
                    f'<div style="font-size:12px;color:{DIM};text-transform:uppercase;'
                    f'letter-spacing:0.5px;margin-bottom:4px;">'
                    f'Turbine Injection Control Range (%)</div>'
                )
                with gr.Row():
                    tic_min_slider = gr.Slider(
                        minimum=tic_min, maximum=tic_max,
                        value=tic_min, label="Min TIC",
                        interactive=True, step=0.1,
                    )
                    tic_max_slider = gr.Slider(
                        minimum=tic_min, maximum=tic_max,
                        value=tic_max, label="Max TIC",
                        interactive=True, step=0.1,
                    )
            apply_btn = gr.Button("Apply Filters", variant="primary", scale=0)

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

        outputs = [
            sensor_out, health_out, eff_out,
            decay_plot, innov_plot, comp_gauge, turb_gauge,
            fuel_plot, temp_plot, press_plot, prop_plot,
        ]
        inputs = [speed_dd, tic_min_slider, tic_max_slider]

        apply_btn.click(fn=update_all, inputs=inputs, outputs=outputs)
        speed_dd.change(fn=update_all, inputs=inputs, outputs=outputs)
        tic_min_slider.change(fn=update_all, inputs=inputs, outputs=outputs)
        tic_max_slider.change(fn=update_all, inputs=inputs, outputs=outputs)
        demo.load(fn=update_all, inputs=inputs, outputs=outputs)

    print("\n--- Launching Digital Twin Dashboard ---")
    print(" Note: Dashboard analyses historical CSV data (not live telemetry)")
    demo.launch(server_name="127.0.0.1", server_port=7860)