import os
import math
import ollama
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# ---------- Config ----------
# Use env override if present; default to the model you actually have installed.
MODEL_NAME = os.getenv("OLLAMA_MODEL", "qwen3:8b")

# Keep prompt small for latency
MAX_HISTORY_MESSAGES = int(os.getenv("CHAT_MAX_HISTORY_MESSAGES", "8"))

# Optional: include expensive diagnostics in chat context (can slow replies)
INCLUDE_DIAGNOSTICS = os.getenv("CHAT_INCLUDE_DIAGNOSTICS", "0") == "1"

# Cache analytics snapshot per dataframe "fingerprint"
_SNAPSHOT_CACHE = {}
_MAX_SNAPSHOT_CACHE = 8

# Cache available model names to avoid repeated ollama.list() calls
_AVAILABLE_MODELS_CACHE = None


# ---------- Utilities ----------
def _safe_float(value, default=np.nan):
    try:
        return float(value)
    except Exception:
        return default


def _first_existing_col(df, candidates, default=np.nan):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _df_fingerprint(df: pd.DataFrame):
    """Cheap-ish fingerprint to reuse cached analytics when the filtered dataframe hasn't changed."""
    if df is None or len(df) == 0:
        return ("empty", 0)

    cols = tuple(df.columns.tolist())
    key_cols = [c for c in ["index", "ship_speed", "tic", "comp_decay", "turb_decay", "gt_torque", "fuel_flow"] if c in df.columns]

    # Use head/tail values only (fast, avoids hashing entire dataframe)
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
    Supports Gradio message-style history:
      [{"role":"user","content":"..."}, {"role":"assistant","content":"..."}]
    and tuple-style history:
      [("user msg", "assistant msg"), ...]
    Returns message-style list.
    """
    if not history:
        return []

    # Normalize tuple-style -> message-style
    if isinstance(history, list) and history and isinstance(history[0], (tuple, list)):
        normalized = []
        for pair in history[-max_messages:]:
            if len(pair) >= 1 and pair[0] is not None:
                normalized.append({"role": "user", "content": str(pair[0])})
            if len(pair) >= 2 and pair[1] is not None:
                normalized.append({"role": "assistant", "content": str(pair[1])})
        return normalized[-max_messages:]

    # Already message-style
    normalized = []
    for msg in history:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content")
        if role in {"user", "assistant", "system"} and content is not None:
            normalized.append({"role": role, "content": str(content)})

    # Keep only recent user/assistant turns (system will be rebuilt each request)
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
            # Ollama versions differ: some use 'name', others include 'model'
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
        # Could not query Ollama list; don't hard-fail here because chat may still work
        return

    if model_name not in names:
        installed = ", ".join(sorted(names)) if names else "(none)"
        raise RuntimeError(
            f"Ollama model '{model_name}' not found. Installed models: {installed}. "
            f"Set OLLAMA_MODEL to an installed model (e.g. qwen3:8b) or run: ollama pull {model_name}"
        )


def _estimate_remaining_life_local(slope, current_health, threshold=0.95):
    """Fallback RUL estimator in samples."""
    try:
        slope = float(slope)
        current_health = float(current_health)
    except Exception:
        return "Unknown"

    if not np.isfinite(slope) or not np.isfinite(current_health):
        return "Unknown"

    # If not degrading (or improving), RUL is not meaningful
    if slope >= 0:
        return "Stable"

    samples_left = (threshold - current_health) / slope  # slope is negative
    if not np.isfinite(samples_left):
        return "Unknown"
    if samples_left <= 0:
        return "Maintenance Due"

    return f"{int(round(samples_left))} samples"


def _estimate_rul(slope, current_health):
    """Try project function first, fallback to local estimate."""
    try:
        from models.DigitalTwin import estimate_remaining_life  # project-specific
        return estimate_remaining_life(slope, current_health)
    except Exception:
        return _estimate_remaining_life_local(slope, current_health)


# ---------- Snapshot / Context ----------
def compute_chat_snapshot(df, dt_twin=None):
    """
    Compute a compact analytics snapshot from the current filtered dataframe.
    This should be reused across chat turns for speed.
    """
    if df is None or len(df) == 0:
        return {
            "ok": False,
            "summary": "No data available for the current filter.",
            "metrics": {},
            "warnings": ["No rows matched the selected filter."],
        }

    # Cache per dataframe fingerprint + whether diagnostics are enabled
    fp = (_df_fingerprint(df), bool(dt_twin), INCLUDE_DIAGNOSTICS)
    if fp in _SNAPSHOT_CACHE:
        return _SNAPSHOT_CACHE[fp]

    latest = df.iloc[-1]

    # Prefer actual sequence column for trends if present
    x = df["index"].to_numpy(dtype=float) if "index" in df.columns else np.arange(len(df), dtype=float)

    # Required-ish columns
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
            "warnings": ["Need at least 2 valid samples for compressor and turbine decay."],
        }
        _cache_snapshot(fp, snapshot)
        return snapshot

    # Align x to the valid rows if there are NaNs
    comp_mask = pd.to_numeric(df["comp_decay"], errors="coerce").notna().to_numpy()
    turb_mask = pd.to_numeric(df["turb_decay"], errors="coerce").notna().to_numpy()
    x_comp = x[comp_mask]
    x_turb = x[turb_mask]

    comp_val = float(comp_series.mean())
    turb_val = float(turb_series.mean())

    # Trends
    comp_slope = float(np.polyfit(x_comp, comp_series.to_numpy(dtype=float), 1)[0]) if len(comp_series) >= 2 else np.nan
    turb_slope = float(np.polyfit(x_turb, turb_series.to_numpy(dtype=float), 1)[0]) if len(turb_series) >= 2 else np.nan

    comp_rul = _estimate_rul(comp_slope, comp_val)
    turb_rul = _estimate_rul(turb_slope, turb_val)

    # Basic status aligned to dashboard semantics (healthy/warn/critical)
    def status(v):
        if not np.isfinite(v):
            return "UNKNOWN"
        if v >= 0.975:
            return "HEALTHY"
        return "CRITICAL"

    # Latest sensor readings (if present)
    latest_metrics = {}
    for key in ["ship_speed", "gt_torque", "fuel_flow", "tic", "gt_rpm", "gg_rpm", "t48", "t2", "p2", "p48"]:
        if key in df.columns:
            latest_metrics[key] = _safe_float(latest.get(key))

    # Optional maintenance history summary
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

    # Optional diagnostics (expensive)
    diag_summary = "Diagnostics disabled for chat speed."
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
                diag_summary = "Top sensor deviations vs baseline:\n- " + "\n- ".join(diag_lines) if diag_lines else "Diagnostics returned no numeric deviations."
            else:
                diag_summary = "Diagnostics returned no deviations."
        except Exception as e:
            diag_summary = f"Diagnostics unavailable ({e})."

    # Concise context block (small prompt = faster)
    latest_ship_speed = latest_metrics.get("ship_speed", np.nan)
    latest_torque = latest_metrics.get("gt_torque", np.nan)
    latest_fuel = latest_metrics.get("fuel_flow", np.nan)
    latest_tic = latest_metrics.get("tic", np.nan)

    summary = (
        "Current filtered propulsion snapshot:\n"
        f"- Samples: {len(df)}\n"
        f"- Latest ship speed: {_fmt(latest_ship_speed, 1)} knots\n"
        f"- Latest GT torque: {_fmt(latest_torque, 1)} kN·m\n"
        f"- Latest fuel flow: {_fmt(latest_fuel, 4)} kg/s\n"
        f"- Latest TIC: {_fmt(latest_tic, 2)}\n"
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
    # Simple FIFO trim
    if len(_SNAPSHOT_CACHE) > _MAX_SNAPSHOT_CACHE:
        oldest_key = next(iter(_SNAPSHOT_CACHE))
        _SNAPSHOT_CACHE.pop(oldest_key, None)


def _build_system_prompt(snapshot):
    context = snapshot.get("summary", "No context available.")
    return (
        "You are a marine propulsion digital twin assistant for engineers. "
        "Be concise, technical, and data-grounded. "
        "Use only the provided snapshot. "
        "If the user asks for something not in the snapshot, say what is missing. "
        "Prefer actionable recommendations. "
        "Use the dashboard threshold semantics: HEALTHY >= 0.95, WARNING 0.90-0.95, CRITICAL < 0.90.\n\n"
        f"{context}"
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


# Disable qwen3 thinking mode — it generates a hidden chain-of-thought before
# every reply, adding seconds of latency even for trivial questions.
_THINK_KWARG = {"think": False} if MODEL_NAME.startswith("qwen3") else {}


# ---------- Public API ----------
def get_system_context(df, dt_twin=None):
    """
    Backward-compatible wrapper.
    Returns a context string, but now uses cached snapshot logic under the hood.
    """
    snapshot = compute_chat_snapshot(df, dt_twin)
    return snapshot.get("summary", "No data available.")


def respond_streaming(message, history, df, dt_twin=None):
    """
    Streaming generator for Gradio chatbot.
    Yields progressively longer assistant text strings as tokens arrive,
    so the UI updates in real time instead of waiting for the full reply.

    Optimisations vs the old blocking respond():
      - stream=True  → first token appears within ~1 s
      - think=False  → suppresses qwen3 chain-of-thought (5-30× speedup)
      - snapshot is cached across turns for the same filtered dataframe
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
    """
    Optional: call once at app startup or first chat tab open.
    Helps catch missing-model errors early and can reduce first-response delay.
    """
    _assert_model_available(MODEL_NAME)
    # Tiny call to warm caches/model
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