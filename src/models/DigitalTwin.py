import pandas as pd
import numpy as np


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

class KalmanFilter:
    """
    A univariate (1D) state-space Kalman filter for tracking decay coefficients.
    """
    
    pass

def compute_kalman_metrics(df):
    """
    Compute psuedo-Kalman innovation metrics from the dataset.
    """
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
    """
    Flag samples where innovation exceeds threshold
    """
    metrics = compute_kalman_metrics(df)
    flags = []
    for label in ["Compressor", "Turbine"]:
        m = metrics[label]
        innov = m["innovations"]
        sigma = m["std_innov"] if m["std_innov"] > 0 else 1e-9
        fault_mask = np.abs(innov) > threshold_sigma * sigma
        n_faults = int(np.sum(fault_mask))
        pct = 100 * n_faults / len(innov) if len(innov) > 0 else 0
        flags.append({
            "component": label, "n_faults": n_faults, "pct": pct,
            "sigma": sigma, "threshold": threshold_sigma,
        })
    return flags, metrics

def estimate_remaining_life(slope, current, threshold=0.90):
    """
    Estimate remaining useful life based on linear degradation trend.
    slope: degradation rate per sample
    current: current health coefficient
    threshold: health level at which maintenance is required
    """
    if slope >= 0:
        return "Stanle" # No degradation or improving
    remaining = (threshold - current) / slope
    return f"{abs(remaining):,.0f} samples" if remaining > 0 else "Maintenance Due"