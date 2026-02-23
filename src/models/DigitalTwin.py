import pandas as pd
import numpy as np


class PropulsionDigitalTwin:
    """
    A Digital Twin for Marine Propulsion Systems.
    Integrates predictive models with health monitoring and maintenance alerts.
    """

    def __init__(self, compressor_model, turbine_model, scaler=None,
                 comp_scaler=None, turb_scaler=None):
        self.compressor_model = compressor_model
        self.turbine_model = turbine_model
        self.scaler = scaler
        self.comp_scaler = comp_scaler
        self.turb_scaler = turb_scaler
        self.health_history = []
        self.MAINTENANCE_THRESHOLD = 0.975 # Updated based on observed data
        
        # Cycle Tracking
        self.maintenance_history = []
        self.current_cycle_start_idx = 0
        self.last_health = {"compressor": 1.0, "turbine": 1.0}

    def predict_health(self, input_data):
        """
        Predicts health from actual sensor readings.
        """
        # Ensure input is 2D
        if len(input_data.shape) == 1:
            input_data = input_data.values.reshape(1, -1)
        else:
            input_data = input_data.values

        X_comp = self.comp_scaler.transform(input_data) if self.comp_scaler else input_data
        X_turb = self.turb_scaler.transform(input_data) if self.turb_scaler else input_data
            
        comp_decay = float(self.compressor_model.predict(X_comp)[0])
        turb_decay = float(self.turbine_model.predict(X_turb)[0])
        
        # Detect Maintenance Events (Jumps)
        self._detect_jumps(comp_decay, turb_decay)

        status = {
            "compressor_health": round(comp_decay, 4),
            "turbine_health": round(turb_decay, 4),
            "comp_alert": comp_decay < self.MAINTENANCE_THRESHOLD,
            "turb_alert": turb_decay < self.MAINTENANCE_THRESHOLD
        }
        
        self.health_history.append(status)
        self.last_health = {"compressor": comp_decay, "turbine": turb_decay}
        return status

    def _detect_jumps(self, current_comp, current_turb):
        """Detect if health jumped back up (Maintenance Event)."""
        # Using a threshold of +0.01 to identify a significant maintenance action
        jump_threshold = 0.01 
        
        if (current_comp > self.last_health["compressor"] + jump_threshold) or \
           (current_turb > self.last_health["turbine"] + jump_threshold):
            
            cycle_duration = len(self.health_history) - self.current_cycle_start_idx
            
            event = {
                "type": "Maintenance Event Detected",
                "sample_index": len(self.health_history),
                "duration": cycle_duration,
                "comp_recovery": round(current_comp, 4),
                "turb_recovery": round(current_turb, 4),
                "effectiveness": "Full" if current_comp > 0.99 else "Partial"
            }
            
            self.maintenance_history.append(event)
            self.current_cycle_start_idx = len(self.health_history)

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

    def run_what_if(self, base_data, adjustments):
        """
        Simulate a 'What-If' scenario.
        base_data: A single row DataFrame/Series of current sensor data
        adjustments: dict of {column_name: new_value}
        """
        sim_data = base_data.copy()
        for col, val in adjustments.items():
            if col in sim_data:
                sim_data[col] = val
        
        # Ensure we have a DataFrame with correct columns for scaling/prediction
        if isinstance(sim_data, pd.Series):
            sim_df = pd.DataFrame([sim_data])
        else:
            sim_df = sim_data

        # We need FEATURE_COLS usually defined in app.py. 
        # For now, assume sim_df has them.
        return self.predict_health(sim_df)

    def diagnose_issues(self, df):
        """
        Identify which sensors are deviating most from expected baseline.
        Simple Root Cause Analysis.
        """
        latest = df.iloc[-1]
        # Basic logic: compare current readings to average of 'healthy' data (health > 0.99)
        healthy_data = df[df["comp_decay"] > 0.99]
        if len(healthy_data) < 5:
            healthy_data = df.iloc[:10] # Fallback to start of dataset

        baseline = healthy_data.mean()
        deviations = {}
        
        # Check key sensors for deviation
        key_sensors = ["t48", "t2", "p48", "p2", "fuel_flow"]
        for s in key_sensors:
            if s in latest and s in baseline:
                diff_pct = (latest[s] - baseline[s]) / baseline[s] * 100
                deviations[s] = diff_pct
        
        return deviations

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
    Flag samples where innovation exceeds threshold.
    Returns (flags, metrics) tuple.
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
            "component": label,
            "n_faults": n_faults,
            "pct": pct,
            "sigma": sigma,
            "threshold": threshold_sigma,
        })
    return flags, metrics

def estimate_remaining_life(slope, current, threshold=0.975):
    """
    Estimate remaining useful life based on linear degradation trend.
    slope: degradation rate per sample
    current: current health coefficient
    threshold: health level at which maintenance is required (default 0.975 based on data)
    """
    if slope >= 0:
        return "Stable" # No degradation or improving
    remaining = (threshold - current) / slope
    return f"{abs(remaining):,.0f} samples" if remaining > 0 else "Maintenance Due"