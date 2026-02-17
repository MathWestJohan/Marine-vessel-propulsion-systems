import pandas as pd
import numpy as np
import joblib
import os
import gradio as gr
from datetime import datetime


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
        # Thresholds for predictive maintenance alerts
        self.MAINTENANCE_THRESHOLD = 0.96

    def predict_health(self, input_data):
        """
        Simulates real-time sensor ingestion and health prediction.
        """
        if self.scaler:
            input_scaled = self.scaler.transform(input_data)
        else:
            input_scaled = input_data

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
            recommendations.append("SCHEDULE COMPRESSOR CLEANING: Efficiency below threshold.")
        if status["turb_alert"]:
            recommendations.append("SCHEDULE TURBINE INSPECTION: Thermal decay detected.")

        if not recommendations:
            return "Systems Operational: No maintenance required."
        return "\n".join(recommendations)


def launch_digital_twin_dashboard(dt_instance):
    """
    Creates a Gradio interface for the Digital Twin.
    """

    def monitor_system(lever_pos, speed, torque, rpm_gt, rpm_gg, prop_torque_s, 
                      prop_torque_p, temp_t48, temp_t2, pressure_p48, 
                      pressure_p2, pressure_pexh, tic, fuel_flow):
        # Define feature names matching the EXACT training data format
        feature_names = [
            'Lever position',
            'Ship speed (v)',
            'Gas Turbine (GT) shaft torque (GTT) (kN m)',
            'GT rate of revolutions (GTn) (rpm)',
            'Gas Generator rate of revolutions (GGn) (rpm)',
            'Starboard Propeller Torque (Ts) (kN)',
            'Port Propeller Torque (Tp) (kN)',
            'Hight Pressure (HP) Turbine exit temperature (T48) (C)',
            'GT Compressor outlet air temperature (T2) (C)',
            'HP Turbine exit pressure (P48) (bar)',
            'GT Compressor outlet air pressure (P2) (bar)',
            'GT exhaust gas pressure (Pexh) (bar)',
            'Turbine Injecton Control (TIC) (%)',
            'Fuel flow (mf) (kg/s)'
        ]
        
        # Create DataFrame with all 14 features from user inputs
        data = pd.DataFrame(
            [[lever_pos, speed, torque, rpm_gt, rpm_gg, prop_torque_s, prop_torque_p,
              temp_t48, temp_t2, pressure_p48, pressure_p2, pressure_pexh, tic, fuel_flow]],
            columns=feature_names
        )
        
        status = dt_instance.predict_health(data)
        recommendation = dt_instance.get_maintenance_recommendation(status)

        health_metrics = f"Compressor Coeff: {status['compressor_health']}\nTurbine Coeff: {status['turbine_health']}"
        return health_metrics, recommendation

    interface = gr.Interface(
        fn=monitor_system,
        inputs=[
            gr.Slider(1, 10, label="Lever Position", value=5),
            gr.Slider(1, 30, label="Ship Speed (v)", value=15),
            gr.Number(label="GT Shaft Torque (kN m)", value=20000),
            gr.Number(label="GT RPM (GTn)", value=2000),
            gr.Number(label="Gas Generator RPM (GGn)", value=8000),
            gr.Number(label="Starboard Prop Torque (kN)", value=200),
            gr.Number(label="Port Prop Torque (kN)", value=200),
            gr.Number(label="HP Turbine Temp (T48) °C", value=700),
            gr.Number(label="Compressor Outlet Temp (T2) °C", value=650),
            gr.Number(label="HP Turbine Pressure (P48) bar", value=2.0),
            gr.Number(label="Compressor Outlet Pressure (P2) bar", value=11.0),
            gr.Number(label="Exhaust Gas Pressure (Pexh) bar", value=1.025),
            gr.Number(label="Turbine Injection Control (%)", value=25),
            gr.Number(label="Fuel Flow (kg/s)", value=0.5)
        ],
        outputs=[
            gr.Textbox(label="Digital Twin Health Metrics"),
            gr.Textbox(label="Predictive Maintenance Dashboard")
        ],
        title="Marine Propulsion Digital Twin",
        description="Real-time monitoring and predictive maintenance dashboard."
    )
    interface.launch()