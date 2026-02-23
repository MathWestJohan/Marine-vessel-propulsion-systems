import os
import ollama
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# Default Ollama model
MODEL_NAME = "llama3"

def get_system_context(df, dt_twin=None):
    """Generate a context string from the current dataframe and digital twin for the LLM."""
    if df is None or len(df) == 0:
        return "No data available."
    
    latest = df.iloc[-1]
    comp_val = df["comp_decay"].mean()
    turb_val = df["turb_decay"].mean()
    
    # RUL (Remaining Useful Life) Calculation
    comp_trend = np.polyfit(range(len(df)), df["comp_decay"].values, 1)
    turb_trend = np.polyfit(range(len(df)), df["turb_decay"].values, 1)
    
    from models.DigitalTwin import estimate_remaining_life
    comp_rul = estimate_remaining_life(comp_trend[0], comp_val)
    turb_rul = estimate_remaining_life(turb_trend[0], turb_val)

    # Maintenance History Analysis
    maint_text = ""
    if dt_twin and hasattr(dt_twin, 'maintenance_history'):
        history = dt_twin.maintenance_history
        if history:
            maint_text = f"\nMaintenance History ({len(history)} events detected):\n"
            for event in history[-3:]: # Show last 3 events
                maint_text += f"- Sample {event['sample_index']}: {event['effectiveness']} recovery (C:{event['comp_recovery']}, T:{event['turb_recovery']}). Cycle duration: {event['duration']} samples.\n"
        else:
            maint_text = "\nMaintenance History: No events detected in this window.\n"

    # Root Cause Analysis (Diagnosis)
    diagnosis_text = ""
    if dt_twin:
        deviations = dt_twin.diagnose_issues(df)
        diagnosis_text = "\nSensor Deviations from Baseline:\n"
        for sensor, dev in deviations.items():
            diagnosis_text += f"- {sensor}: {dev:+.2f}%\n"

    context = f"""
    Current Vessel State:
    - Ship Speed: {latest['ship_speed']:.1f} knots
    - GT Shaft Torque: {latest['gt_torque']:.1f} kN m
    - Fuel Flow: {latest['fuel_flow']:.4f} kg/s
    
    Health Monitoring:
    - Compressor Health: {comp_val:.4f} (Est. RUL: {comp_rul})
    - Turbine Health: {turb_val:.4f} (Est. RUL: {turb_rul})
    {maint_text}
    {diagnosis_text}
    
    Degradation Rate (Slope):
    - Compressor: {comp_trend[0]:.2e} per sample
    - Turbine: {turb_trend[0]:.2e} per sample
    """
    return context

def respond(message, history, df, dt_twin=None):
    """Chat response function for Gradio."""
    context = get_system_context(df, dt_twin)
    
    system_instruction = f"""
    You are an AI Assistant for a Marine Propulsion Digital Twin. 
    You help engineers monitor gas turbine health, diagnose root causes, and analyze maintenance cycles.
    
    {context}
    
    Capabilities:
    1. Maintenance Analysis: Identify if recent maintenance was 'Full' or 'Partial'. If 'Partial', warn that fouling may be permanent.
    2. Degradation Rates: Compare the 'Slope' values. If the current slope is steeper than usual, suggest that the engine is aging faster.
    3. Failure Prediction: We use 0.975 as the critical maintenance threshold. Advise based on the 'Est. RUL'.
    
    Guidelines:
    - Be professional, technical, and data-driven.
    - Mention the 'Sawtooth' pattern if you see multiple maintenance events.
    - If health is below 0.975, recommend immediate intervention.
    """
    
    messages = [{"role": "system", "content": system_instruction}]
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    
    messages.append({"role": "user", "content": message})
    
    try:
        response = ollama.chat(model=MODEL_NAME, messages=messages)
        assistant_message = response['message']['content']
        
        new_history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": assistant_message}
        ]
        return new_history
    except Exception as e:
        return history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": f"Error: {str(e)}"}
        ]
