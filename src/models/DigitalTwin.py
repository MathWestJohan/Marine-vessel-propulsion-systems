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
            "timestamp": datetime.now().strftime("%H:%M:%S"),
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
            recommendations.append("⚠️ SCHEDULE COMPRESSOR CLEANING: Efficiency below threshold.")
        if status["turb_alert"]:
            recommendations.append("⚠️ SCHEDULE TURBINE INSPECTION: Thermal decay detected.")

        if not recommendations:
            return "✅ Systems Operational: No maintenance required."
        return "\n".join(recommendations)


def launch_digital_twin_dashboard(dt_instance):
    """
    Creates a Gradio interface for the Digital Twin.
    """

    def monitor_system(speed, torque, fuel_flow, lever_pos):
        # Create a sample input matching the feature structure
        # (Assuming a simplified feature set for the GUI)
        data = np.array([[lever_pos, speed, torque, fuel_flow] + [0] * 12])  # Padding to match training features
        status = dt_instance.predict_health(data)

        recommendation = dt_instance.get_maintenance_recommendation(status)

        health_metrics = f"Compressor Coeff: {status['compressor_health']}\nTurbine Coeff: {status['turbine_health']}"
        return health_metrics, recommendation

    interface = gr.Interface(
        fn=monitor_system,
        inputs=[
            gr.Slider(1, 30, label="Ship Speed (v)"),
            gr.Number(label="GT Shaft Torque (GTT)"),
            gr.Number(label="Fuel Flow (mf)"),
            gr.Slider(1, 10, label="Lever Position")
        ],
        outputs=[
            gr.Textbox(label="Digital Twin Health Metrics"),
            gr.Textbox(label="Predictive Maintenance Dashboard")
        ],
        title="Marine Propulsion Digital Twin",
        description="Real-time monitoring and predictive maintenance dashboard."
    )
    interface.launch()