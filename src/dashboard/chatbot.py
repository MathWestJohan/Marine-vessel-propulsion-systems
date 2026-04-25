import ollama
import pandas as pd

# The model you have pulled in Ollama (e.g., 'llama3', 'mistral', 'phi3')
OLLAMA_MODEL = "llama3"

def get_system_context(df_snapshot, comp_health, turb_health, comp_rul, turb_rul, active_faults="None"):
    """
    Generates a dynamic system prompt based on the current dashboard state.
    This gives the LLM real-time knowledge of the engine's health.
    """
    context = f"""
You are the AI Chief Engineer of a marine vessel. You are monitoring the health of a Gas Turbine and Compressor system.
Your job is to provide concise, technical, and professional answers based on the current telemetry data.

CURRENT SYSTEM STATE:
- Ship Speed: {df_snapshot['ship_speed']:.0f} knots
- Gas Turbine Health Index: {turb_health:.4f} (Threshold: 0.975) -> RUL: {turb_rul}
- Compressor Health Index: {comp_health:.4f} (Threshold: 0.950) -> RUL: {comp_rul}

KEY SENSOR READINGS:
- GT Torque: {df_snapshot['gt_torque']:,.0f} kN m
- GT RPM: {df_snapshot['gt_rpm']:,.0f}
- Fuel Flow: {df_snapshot['fuel_flow']:.4f} kg/s
- Turbine Exit Temp (T48): {df_snapshot['t48']:.1f} °C
- Compressor Outlet Pres (P2): {df_snapshot['p2']:.3f} bar

ACTIVE FAULTS INJECTED: {active_faults}

RULES:
1. If health is below the threshold, warn the user.
2. Be direct and analytical. Avoid generic AI disclaimers.
3. If asked about a sensor, reference the exact value provided above.
"""
    return context

def chat_stream(message, history, df_snapshot, comp_health, turb_health, comp_rul, turb_rul, active_faults_dict):
    """
    Streams the response from the local Ollama model to the Gradio UI.
    """
    if df_snapshot is None or df_snapshot.empty:
        yield "Error: No data loaded. Please select a mission cycle first."
        return

    # Format active faults for the prompt
    fault_str = "None"
    if active_faults_dict:
        fault_str = ", ".join([f"{k} ({v:+.1f}%)" for k, v in active_faults_dict.items()])

    # Build the system prompt
    system_prompt = get_system_context(df_snapshot, comp_health, turb_health, comp_rul, turb_rul, fault_str)

    # Construct the conversation history for Ollama
    messages = [{"role": "system", "content": system_prompt}]
    
    # Handle Gradio history format (list of dicts with 'role' and 'content')
    for turn in history:
        messages.append(turn)
        
    messages.append({"role": "user", "content": message})

    try:
        # Stream the response from the local model
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=messages,
            stream=True
        )
        
        partial_message = ""
        for chunk in response:
            if 'message' in chunk and 'content' in chunk['message']:
                partial_message += chunk['message']['content']
                yield partial_message
                
    except Exception as e:
        yield f"Error communicating with Ollama. Make sure Ollama is running and you have pulled the '{OLLAMA_MODEL}' model. (Error: {str(e)})"
