import os
import math
import ollama
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# Config
MODEL_NAME = os.getenv("OLLAMA_MODEL", "qwen3:8b")
MAX_HISTORY_MESSAGES = int(os.getenv("CHAT_MAX_HISTORY_MESSAGES", "8"))

# Diagnostics enabled by default — the latency cost is minimal compared
# to LLM inference, and the root cause data makes responses substantive.
INCLUDE_DIAGNOSTICS = os.getenv("CHAT_INCLUDE_DIAGNOSTICS", "1") == "1"

_SNAPSHOT_CACHE = {}
_MAX_SNAPSHOT_CACHE = 8
_AVAILABLE_MODELS_CACHE = None


# Utilities
def _safe_float(value, default=np.nan):
    try:
        return float(value)
    except Exception:
        return default


def _first_existing_col(df, candidates, default=np.nan):
    for c in candidates:
        if c in df.columns:
            return c
    return default


def _df_fingerprint(df: pd.DataFrame):
    """Cheap fingerprint to reuse cached analytics when the filtered dataframe hasn't changed."""
    if df is None or len(df) == 0:
        return ("empty", 0)

    cols = tuple(df.columns.tolist())
    key_cols = [c for c in ["index", "ship_speed", "tic", "comp_decay", "turb_decay", "gt_torque", "fuel_flow"] if c in df.columns]

    sample = pd.concat([df[key_cols].head(2), df[key_cols].tail(2)], axis=0) if key_cols else pd.DataFrame()
    sample_tuple = tuple(sample.fillna("NA").astype(str).to_numpy().flatten().tolist()) if not sample.empty else ()

    return (
        len(df),
        cols,
        sample_tuple,
        str(df.index[0]) if len(df) else None,
        str(df.index[-1]) if len(df) else None,
    )


def _trim_history(history, max_messages=MAX_HISTORY_MESSAGES):
    """
    Supports Gradio message-style history and tuple-style history.
    Returns message-style list.
    """
    if not history:
        return []

    if isinstance(history, list) and history and isinstance(history[0], (tuple, list)):
        normalized = []
        for pair in history[-max_messages:]:
            if len(pair) >= 1 and pair[0] is not None:
                normalized.append({"role": "user", "content": str(pair[0])})
            if len(pair) >= 2 and pair[1] is not None:
                normalized.append({"role": "assistant", "content": str(pair[1])})
        return normalized[-max_messages:]

    normalized = []
    for msg in history:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content")
        if role in {"user", "assistant", "system"} and content is not None:
            normalized.append({"role": role, "content": str(content)})

    normalized = [m for m in normalized if m["role"] in {"user", "assistant"}]
    return normalized[-max_messages:]


def _get_available_models():
    global _AVAILABLE_MODELS_CACHE
    if _AVAILABLE_MODELS_CACHE is not None:
        return _AVAILABLE_MODELS_CACHE

    try:
        res = ollama.list()
        models = res.get("models", []) if isinstance(res, dict) else []
        names = set()
        for m in models:
            if not isinstance(m, dict):
                continue
            if m.get("name"):
                names.add(m["name"])
            if m.get("model"):
                names.add(m["model"])
        _AVAILABLE_MODELS_CACHE = names
    except Exception:
        _AVAILABLE_MODELS_CACHE = set()

    return _AVAILABLE_MODELS_CACHE


def _assert_model_available(model_name: str):
    names = _get_available_models()
    if not names:
        return
    if model_name not in names:
        installed = ", ".join(sorted(names)) if names else "(none)"
        raise RuntimeError(
            f"Ollama model '{model_name}' not found. Installed: {installed}. "
            f"Run: ollama pull {model_name}"
        )


def _estimate_remaining_life_local(slope, current_health, threshold=0.975):
    """Fallback RUL estimator in samples."""
    try:
        slope = float(slope)
        current_health = float(current_health)
    except Exception:
        return "Unknown"

    if not np.isfinite(slope) or not np.isfinite(current_health):
        return "Unknown"
    if slope >= 0:
        return "Stable"

    samples_left = (threshold - current_health) / slope
    if not np.isfinite(samples_left):
        return "Unknown"
    if samples_left <= 0:
        return "Maintenance Due"

    return f"{int(round(samples_left))} samples"


def _estimate_rul(slope, current_health):
    """Try project function first, fallback to local estimate."""
    try:
        from models.DigitalTwin import estimate_remaining_life
        return estimate_remaining_life(slope, current_health)
    except Exception:
        return _estimate_remaining_life_local(slope, current_health)


# Domain Knowledge for System Prompt
DOMAIN_KNOWLEDGE = """
DOMAIN KNOWLEDGE — Marine Gas Turbine Propulsion:

Sensor definitions:
- T48 (HP Turbine exit temperature): Exhaust gas temperature leaving the high-pressure turbine stage. Rising T48 indicates the turbine is working harder, often compensating for upstream compressor degradation.
- T1 (Compressor inlet air temperature): Ambient air temperature entering the compressor. Affects mass flow and compressor efficiency.
- T2 (Compressor outlet air temperature): Air temperature after compression. Abnormally high T2 suggests compressor inefficiency or fouling.
- P48 (HP Turbine exit pressure): Pressure at the turbine exhaust. Drops may indicate blade erosion or seal leakage.
- P1 (Compressor inlet air pressure): Ambient intake pressure. Normally stable unless intake is obstructed.
- P2 (Compressor outlet air pressure): Compressed air delivery pressure. Drops here indicate compressor fouling or blade damage.
- Pexh (Exhaust gas pressure): Back-pressure in the exhaust system. Elevated values suggest exhaust path obstruction.
- Fuel flow (mf): Fuel consumption rate in kg/s. Rising fuel flow at constant speed/torque indicates efficiency loss.
- GT shaft torque (GTT): Output torque of the gas turbine in kN·m.
- TIC (Turbine Injection Control): Control signal for fuel injection in percentage.

Degradation interpretation:
- Decay coefficients represent component health as a fraction of nominal (1.0 = perfect).
- Values above 0.975 are HEALTHY. Below 0.975 is CRITICAL (approx. 2.5% degradation).
- A drop of 5-10% in efficiency is considered critical for gas turbines and typically requires maintenance intervention.

Cross-sensor patterns:
- T48 rising + comp_decay dropping = turbine compensating for compressor fouling. The turbine runs hotter to maintain output power despite reduced compressor efficiency.
- Fuel flow rising + torque stable = overall thermodynamic efficiency loss. More fuel is needed for the same mechanical output.
- P2 dropping + T2 rising = compressor fouling. Fouled blades reduce pressure ratio while friction generates excess heat.
- P48 dropping + T48 rising = possible turbine blade erosion or seal leakage. Gas energy is lost before the turbine can extract it.

Maintenance events:
- Full recovery (coefficient returns to >0.99): Successful overhaul or blade replacement.
- Partial recovery (coefficient improves but stays below 0.99): Cleaning/washing performed but permanent degradation remains (erosion, pitting).
- Shortened maintenance cycles indicate accelerating degradation — may require root cause investigation rather than routine servicing.

Response format:
Always structure your responses as:
1. STATUS: State whether each component is Healthy or Critical with current values.
2. KEY FINDINGS: Summarize the most important observations from the data.
3. RECOMMENDED ACTIONS: Provide specific, actionable maintenance recommendations.
4. REASONING: Explain the technical basis for your recommendations using the sensor data and cross-sensor patterns described above.
"""


# Snapshot / Context
def compute_chat_snapshot(df, dt_twin=None):
    """
    Compute a compact analytics snapshot from the current filtered dataframe.
    """
    if df is None or len(df) == 0:
        return {
            "ok": False,
            "summary": "No data available for the current filter.",
            "metrics": {},
            "warnings": ["No rows matched the selected filter."],
        }

    fp = (_df_fingerprint(df), bool(dt_twin), INCLUDE_DIAGNOSTICS)
    if fp in _SNAPSHOT_CACHE:
        return _SNAPSHOT_CACHE[fp]

    latest = df.iloc[-1]
    x = df["index"].to_numpy(dtype=float) if "index" in df.columns else np.arange(len(df), dtype=float)

    if "comp_decay" not in df.columns or "turb_decay" not in df.columns:
        snapshot = {
            "ok": False,
            "summary": "Filtered data is missing comp_decay and/or turb_decay columns.",
            "metrics": {},
            "warnings": ["Missing target columns for health context."],
        }
        _cache_snapshot(fp, snapshot)
        return snapshot

    comp_series = pd.to_numeric(df["comp_decay"], errors="coerce").dropna()
    turb_series = pd.to_numeric(df["turb_decay"], errors="coerce").dropna()

    if len(comp_series) < 2 or len(turb_series) < 2:
        snapshot = {
            "ok": False,
            "summary": "Not enough samples to compute trends reliably.",
            "metrics": {},
            "warnings": ["Need at least 2 valid samples."],
        }
        _cache_snapshot(fp, snapshot)
        return snapshot

    comp_mask = pd.to_numeric(df["comp_decay"], errors="coerce").notna().to_numpy()
    turb_mask = pd.to_numeric(df["turb_decay"], errors="coerce").notna().to_numpy()
    x_comp = x[comp_mask]
    x_turb = x[turb_mask]

    comp_val = float(comp_series.mean())
    turb_val = float(turb_series.mean())

    comp_slope = float(np.polyfit(x_comp, comp_series.to_numpy(dtype=float), 1)[0]) if len(comp_series) >= 2 else np.nan
    turb_slope = float(np.polyfit(x_turb, turb_series.to_numpy(dtype=float), 1)[0]) if len(turb_series) >= 2 else np.nan

    comp_rul = _estimate_rul(comp_slope, comp_val)
    turb_rul = _estimate_rul(turb_slope, turb_val)

    def status(v):
        if not np.isfinite(v):
            return "UNKNOWN"
        if v >= 0.975:
            return "HEALTHY"
        return "CRITICAL"

    latest_metrics = {}
    for key in ["ship_speed", "gt_torque", "fuel_flow", "tic", "gt_rpm", "gg_rpm", "t48", "t2", "t1", "p2", "p48", "pexh"]:
        if key in df.columns:
            latest_metrics[key] = _safe_float(latest.get(key))

    # Maintenance history
    maint_summary = "No maintenance history available."
    if dt_twin is not None and hasattr(dt_twin, "maintenance_history"):
        try:
            mh = getattr(dt_twin, "maintenance_history", []) or []
            if mh:
                lines = []
                for event in mh[-3:]:
                    if not isinstance(event, dict):
                        continue
                    lines.append(
                        f"Sample {event.get('sample_index', '?')}: "
                        f"{event.get('effectiveness', 'Unknown')} recovery "
                        f"(C:{event.get('comp_recovery', '?')}, T:{event.get('turb_recovery', '?')}), "
                        f"duration {event.get('duration', '?')} samples"
                    )
                maint_summary = "Recent maintenance events:\n- " + "\n- ".join(lines) if lines else "Maintenance history present but unreadable."
            else:
                maint_summary = "No maintenance events detected in this window."
        except Exception as e:
            maint_summary = f"Maintenance history unavailable ({e})."

    # Diagnostics (now enabled by default)
    diag_summary = "Diagnostics disabled."
    if INCLUDE_DIAGNOSTICS and dt_twin is not None and hasattr(dt_twin, "diagnose_issues"):
        try:
            deviations = dt_twin.diagnose_issues(df)
            if isinstance(deviations, dict) and deviations:
                top = sorted(
                    deviations.items(),
                    key=lambda kv: abs(float(kv[1])) if _is_number(kv[1]) else -1,
                    reverse=True,
                )[:6]
                diag_lines = [f"{k}: {float(v):+,.2f}%" for k, v in top if _is_number(v)]
                diag_summary = "Top sensor deviations vs healthy baseline:\n- " + "\n- ".join(diag_lines) if diag_lines else "Diagnostics returned no numeric deviations."
            else:
                diag_summary = "Diagnostics returned no deviations."
        except Exception as e:
            diag_summary = f"Diagnostics unavailable ({e})."

    latest_ship_speed = latest_metrics.get("ship_speed", np.nan)
    latest_torque = latest_metrics.get("gt_torque", np.nan)
    latest_fuel = latest_metrics.get("fuel_flow", np.nan)
    latest_tic = latest_metrics.get("tic", np.nan)
    latest_t48 = latest_metrics.get("t48", np.nan)
    latest_t2 = latest_metrics.get("t2", np.nan)
    latest_p2 = latest_metrics.get("p2", np.nan)
    latest_p48 = latest_metrics.get("p48", np.nan)

    summary = (
        "Current filtered propulsion snapshot:\n"
        f"- Samples: {len(df)}\n"
        f"- Latest ship speed: {_fmt(latest_ship_speed, 1)} knots\n"
        f"- Latest GT torque: {_fmt(latest_torque, 1)} kN·m\n"
        f"- Latest fuel flow: {_fmt(latest_fuel, 4)} kg/s\n"
        f"- Latest TIC: {_fmt(latest_tic, 2)}%\n"
        f"- Latest T48: {_fmt(latest_t48, 1)}°C | T2: {_fmt(latest_t2, 1)}°C\n"
        f"- Latest P2: {_fmt(latest_p2, 3)} bar | P48: {_fmt(latest_p48, 3)} bar\n"
        f"- Compressor health (mean): {_fmt(comp_val, 4)} [{status(comp_val)}], slope {_fmt_sci(comp_slope)}/sample, RUL {comp_rul}\n"
        f"- Turbine health (mean): {_fmt(turb_val, 4)} [{status(turb_val)}], slope {_fmt_sci(turb_slope)}/sample, RUL {turb_rul}\n"
        f"- {maint_summary}\n"
        f"- {diag_summary}"
    )

    snapshot = {
        "ok": True,
        "summary": summary,
        "metrics": {
            "samples": len(df),
            "comp_health_mean": comp_val,
            "turb_health_mean": turb_val,
            "comp_slope": comp_slope,
            "turb_slope": turb_slope,
            "comp_rul": comp_rul,
            "turb_rul": turb_rul,
            "latest": latest_metrics,
        },
        "warnings": [],
    }
    _cache_snapshot(fp, snapshot)
    return snapshot


def _cache_snapshot(key, snapshot):
    _SNAPSHOT_CACHE[key] = snapshot
    if len(_SNAPSHOT_CACHE) > _MAX_SNAPSHOT_CACHE:
        oldest_key = next(iter(_SNAPSHOT_CACHE))
        _SNAPSHOT_CACHE.pop(oldest_key, None)


def _build_system_prompt(snapshot):
    context = snapshot.get("summary", "No context available.")
    return (
        "You are a marine propulsion digital twin assistant for engineers monitoring "
        "a gas turbine propulsion system. You are technical, data-grounded, and concise. "
        "Use only the provided snapshot and domain knowledge below. "
        "If the user asks for something not in the snapshot, say what is missing. "
        "Prefer actionable recommendations.\n\n"
        f"{DOMAIN_KNOWLEDGE}\n\n"
        f"CURRENT SYSTEM SNAPSHOT:\n{context}"
    )


def _is_number(v):
    try:
        float(v)
        return True
    except Exception:
        return False


def _fmt(v, decimals=2):
    try:
        if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
            return "N/A"
        return f"{float(v):.{decimals}f}"
    except Exception:
        return "N/A"


def _fmt_sci(v):
    try:
        if v is None or not np.isfinite(float(v)):
            return "N/A"
        return f"{float(v):.2e}"
    except Exception:
        return "N/A"


# Disable qwen3 thinking mode
_THINK_KWARG = {"think": False} if MODEL_NAME.startswith("qwen3") else {}


# Public API
def get_system_context(df, dt_twin=None):
    """Backward-compatible wrapper. Returns a context string."""
    snapshot = compute_chat_snapshot(df, dt_twin)
    return snapshot.get("summary", "No data available.")


def respond_streaming(message, history, df, dt_twin=None):
    """
    Streaming generator for Gradio chatbot.
    Yields progressively longer assistant text strings as tokens arrive.
    """
    normalized_history = _trim_history(history, MAX_HISTORY_MESSAGES)

    try:
        snapshot = compute_chat_snapshot(df, dt_twin)
    except Exception as e:
        snapshot = {
            "ok": False,
            "summary": f"Could not build dashboard context: {e}",
            "metrics": {},
            "warnings": [str(e)],
        }

    system_instruction = _build_system_prompt(snapshot)

    messages = [{"role": "system", "content": system_instruction}]
    messages.extend(normalized_history)
    messages.append({"role": "user", "content": message})

    try:
        _assert_model_available(MODEL_NAME)

        accumulated = ""
        for chunk in ollama.chat(
            model=MODEL_NAME,
            messages=messages,
            stream=True,
            **_THINK_KWARG,
        ):
            accumulated += chunk.get("message", {}).get("content", "")
            yield accumulated

    except Exception as e:
        yield f"Error: {e}"


def respond(message, history, df, dt_twin=None):
    """Non-streaming fallback (used by generate_report and tests)."""
    normalized_history = _trim_history(history, MAX_HISTORY_MESSAGES)

    try:
        snapshot = compute_chat_snapshot(df, dt_twin)
    except Exception as e:
        snapshot = {
            "ok": False,
            "summary": f"Could not build dashboard context: {e}",
            "metrics": {},
            "warnings": [str(e)],
        }

    system_instruction = _build_system_prompt(snapshot)

    messages = [{"role": "system", "content": system_instruction}]
    messages.extend(normalized_history)
    messages.append({"role": "user", "content": message})

    try:
        _assert_model_available(MODEL_NAME)
        response = ollama.chat(
            model=MODEL_NAME,
            messages=messages,
            **_THINK_KWARG,
        )
        assistant_message = response["message"]["content"]
    except Exception as e:
        assistant_message = f"Error: {str(e)}"

    return normalized_history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": assistant_message},
    ]


def warmup_model():
    """Optional: call once at app startup to catch missing-model errors early."""
    _assert_model_available(MODEL_NAME)
    try:
        ollama.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "ping"}],
        )
        return True, f"Chat model '{MODEL_NAME}' is ready."
    except Exception as e:
        return False, f"Chat model warmup failed: {e}"


def clear_chat_snapshot_cache():
    _SNAPSHOT_CACHE.clear()