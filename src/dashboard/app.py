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


def _load_data(data_path=None):
    if data_path is None:
        data_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "data.csv")
    df = pd.read_csv(data_path)
    df.columns = [c.strip() for c in df.columns]
    df.rename(columns=COL_MAP, inplace=True)
    return df


def launch_dashboard(dt_instance=None, data_path=None):
    """Launch the Gradio dashboard."""
    RAW_DF = _load_data(data_path)
    BASELINES = RAW_DF.groupby("ship_speed").first().reset_index()
    SPEED_CHOICES = ["All Speeds"] + [f"{v} kn" for v in sorted(RAW_DF["ship_speed"].unique())]
    lever_choices = ["All"] + [str(v) for v in sorted(RAW_DF["lever_pos"].unique())]
    tic_min = float(RAW_DF["tic"].min())
    tic_max = float(RAW_DF["tic"].max())

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

    def update_all(lever, speed, tic_min_val, tic_max_val):
        df = filter_df(lever, speed, (tic_min_val, tic_max_val))
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

    with gr.Blocks(title="Marine GT Propulsion Monitor") as demo:
        gr.HTML(header_html(len(RAW_DF)))

        gr.HTML(f'<div class="section-label">Input Panel &mdash; Operator Controls</div>')
        with gr.Row():
            lever_dd = gr.Dropdown(choices=lever_choices, value="All",
                                   label="Lever Position (1-10)", interactive=True)
            speed_dd = gr.Dropdown(choices=SPEED_CHOICES, value="All Speeds",
                                   label="Ship Speed Demand", interactive=True)
            with gr.Column():
                gr.HTML(f'<div style="font-size:12px;color:{DIM};text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px;">Turbine Injection Control Range (%)</div>')
                with gr.Row():
                    tic_min_slider = gr.Slider(minimum=tic_min, maximum=tic_max,
                                               value=tic_min, label="Min TIC",
                                               interactive=True, step=0.1)
                    tic_max_slider = gr.Slider(minimum=tic_min, maximum=tic_max,
                                               value=tic_max, label="Max TIC",
                                               interactive=True, step=0.1)
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
        inputs = [lever_dd, speed_dd, tic_min_slider, tic_max_slider]

        apply_btn.click(fn=update_all, inputs=inputs, outputs=outputs)
        lever_dd.change(fn=update_all, inputs=inputs, outputs=outputs)
        speed_dd.change(fn=update_all, inputs=inputs, outputs=outputs)
        tic_min_slider.change(fn=update_all, inputs=inputs, outputs=outputs)
        tic_max_slider.change(fn=update_all, inputs=inputs, outputs=outputs)
        demo.load(fn=update_all, inputs=inputs, outputs=outputs)

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        css=get_css(),
        theme=gr.themes.Base(),
    )