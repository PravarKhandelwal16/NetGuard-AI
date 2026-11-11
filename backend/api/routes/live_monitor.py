"""
NETGUARD AI — Live Website & Service Telemetry Anomaly Detection Route.
Probes live target URLs/services, computes real-time network telemetry
(latency, jitter, status codes, packet sizes), and evaluates multi-signal
anomaly scores with self-tuning adaptive thresholds.
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import time
import requests
import numpy as np

router = APIRouter(prefix="/live", tags=["Live Monitoring"])

# Persistent session with realistic browser headers
session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
})

# In-memory history of probed telemetry per target
history_store: Dict[str, List[Dict[str, Any]]] = {}
attack_simulation_state: Dict[str, Optional[str]] = {}


def normalize_target_url(raw_url: str) -> str:
    url = raw_url.strip()
    if not (url.startswith("http://") or url.startswith("https://")):
        if "localhost" in url or "127.0.0.1" in url:
            url = "http://" + url
        else:
            url = "https://" + url
    return url


class ProbeRequest(BaseModel):
    url: str = Field(..., description="Target website or API endpoint (e.g. https://google.com)")
    simulate_attack: Optional[str] = Field(None, description="Attack simulation mode: 'ddos_spike', 'gateway_error', 'latency_degradation', 'jitter_burst', or None")
    alpha: float = Field(0.5, ge=0.1, le=0.9, description="Dual-signal latency vs variance weight")
    base_threshold: float = Field(32.7688, description="Base threshold tau_0")
    engine_mode: str = Field("stabilized", description="Threshold engine: 'stabilized', 'rolling_mad', or 'fixed'")


@router.post("/probe")
async def probe_target_service(req: ProbeRequest):
    """
    Performs a live network probe against the target URL,
    extracts real-time telemetry, and runs AI anomaly detection.
    Guaranteed to return valid JSON telemetry even on network faults.
    """
    clean_url = normalize_target_url(req.url)
    target_key = clean_url.lower()

    if target_key not in history_store:
        history_store[target_key] = []

    target_history = history_store[target_key]
    sim_mode = req.simulate_attack or attack_simulation_state.get(target_key)

    # 1. Measure Live Network Metrics
    t_start = time.perf_counter()
    status_code = 200
    content_length = 1024
    error_msg = None
    is_fault = False

    try:
        # Use GET with stream=True so headers and initial response stream quickly without downloading full pages
        resp = session.get(clean_url, timeout=2.5, stream=True, allow_redirects=True)
        t_end = time.perf_counter()
        actual_latency_ms = (t_end - t_start) * 1000.0
        status_code = resp.status_code
        content_length = int(resp.headers.get("Content-Length", 2048))
        # Release connection back to pool
        resp.close()
    except requests.exceptions.Timeout:
        t_end = time.perf_counter()
        actual_latency_ms = 2500.0
        status_code = 504
        error_msg = "Gateway Connection Timeout (>2.5s)"
        is_fault = True
    except requests.exceptions.ConnectionError as ce:
        t_end = time.perf_counter()
        actual_latency_ms = min(1200.0, (t_end - t_start) * 1000.0 + 100.0)
        status_code = 502
        error_msg = "Connection Refused / Host Unreachable"
        is_fault = True
    except Exception as e:
        t_end = time.perf_counter()
        actual_latency_ms = (t_end - t_start) * 1000.0
        status_code = 500
        error_msg = f"Network Transport Error: {str(e)[:60]}"
        is_fault = True

    # 2. Apply Attack Simulation if Active (to demonstrate live AI detection)
    effective_latency_ms = actual_latency_ms
    effective_status = status_code

    if sim_mode == "ddos_spike":
        # Volumetric DDoS burst: high latency spike, packet inflation
        effective_latency_ms = actual_latency_ms + float(np.random.uniform(650.0, 980.0))
        content_length = int(content_length * np.random.uniform(4.0, 8.0))
    elif sim_mode == "gateway_error":
        # Upstream service crash: HTTP 502/503
        effective_status = 502
        effective_latency_ms = actual_latency_ms + float(np.random.uniform(300.0, 600.0))
        error_msg = "Simulated Upstream Service Crash"
    elif sim_mode == "latency_degradation":
        # Slowloris / Thread exhaustion
        effective_latency_ms = actual_latency_ms + float(np.random.uniform(400.0, 650.0))
    elif sim_mode == "jitter_burst":
        # Routing instability / packet flapping
        effective_latency_ms = max(25.0, actual_latency_ms + float(np.random.choice([-120.0, 380.0])))

    # 3. Compute Rolling Baseline & Jitter
    lat_arr = np.array([h["latency_ms"] for h in target_history[-50:]], dtype=float)
    if len(lat_arr) >= 5:
        base_median_latency = float(np.median(lat_arr))
        mad_latency = float(np.median(np.abs(lat_arr - base_median_latency))) or 14.0
    else:
        base_median_latency = max(20.0, actual_latency_ms)
        mad_latency = 14.0

    last_latency = target_history[-1]["latency_ms"] if target_history else base_median_latency
    jitter_ms = abs(effective_latency_ms - last_latency)

    # 4. Multi-Signal AI Anomaly Scoring
    # Signal 1: Latency Deviation Score (e_r)
    latency_delta = max(0.0, effective_latency_ms - base_median_latency)
    score_latency = (latency_delta / (mad_latency + 1e-4)) * 12.0

    # Signal 2: Protocol & Status Penalty (d_z)
    if effective_status >= 500 or error_msg:
        score_status = 78.0
    elif effective_status == 429:
        score_status = 58.0
    elif effective_status >= 400:
        score_status = 32.0
    elif jitter_ms > 200.0:
        score_status = 38.0
    else:
        score_status = min(18.0, (jitter_ms / (mad_latency + 1e-4)) * 5.0)

    # Composite Fusion Score: S_t = alpha * score_latency + (1 - alpha) * score_status + baseline
    alpha = req.alpha
    composite_score = float(np.round((alpha * score_latency) + ((1.0 - alpha) * score_status) + 10.0, 2))

    # 5. Multi-Engine Adaptive Thresholding
    tau_0 = req.base_threshold

    # Engine A: Fixed Baseline Threshold (Rigid Static Reference)
    tau_fixed = float(round(tau_0, 2))

    # Engine B: Original Adaptive (Rolling MAD - Naive Unbounded, Runaway Risk)
    all_scores = [h["score"] for h in target_history[-25:]] + [composite_score]
    arr_all = np.array(all_scores, dtype=float)
    med_all = float(np.median(arr_all))
    mad_all = float(np.median(np.abs(arr_all - med_all)))
    tau_rolling_mad = float(np.round(med_all + 2.5 * 1.4826 * max(mad_all, 2.0), 2))

    # Engine C: Stabilized Adaptive Threshold (Bounded Clamping + Attack Quarantine + Exponential Inertia)
    clean_scores = [h["score"] for h in target_history[-25:] if not h.get("is_anomaly", False)]
    if len(clean_scores) >= 3:
        arr_clean = np.array(clean_scores, dtype=float)
        med_clean = float(np.median(arr_clean))
        mad_clean = float(np.median(np.abs(arr_clean - med_clean)))
        proposed_tau = med_clean + 2.5 * 1.4826 * max(mad_clean, 2.0)
        # Bounded clamping strictly within [0.80 * tau_0, 1.25 * tau_0]
        clamped_tau = max(tau_0 * 0.80, min(tau_0 * 1.25, proposed_tau))
        last_tau = target_history[-1].get("threshold_stabilized", target_history[-1].get("threshold", tau_0))
        # Exponential smoothing rate (15% for responsive adaptation to real load fluctuations)
        tau_stabilized = float(np.round(0.15 * clamped_tau + 0.85 * last_tau, 2))
    else:
        tau_stabilized = float(round(tau_0, 2))

    # Select Active Threshold according to User-Configured Engine Mode
    engine = req.engine_mode.lower()
    if "rolling" in engine or "mad" in engine:
        current_threshold = tau_rolling_mad
        adaptation_mode = "Rolling MAD (Unbounded)"
        drift_status = "RUNAWAY INFLATION" if (tau_rolling_mad > tau_0 * 1.3) else "ADAPTING TO STREAM"
    elif "fixed" in engine:
        current_threshold = tau_fixed
        adaptation_mode = "Fixed Baseline"
        drift_status = "STATIC (NO ADAPTATION)"
    else:
        current_threshold = tau_stabilized
        adaptation_mode = "Stabilized Adaptive (Bounded)"
        drift_status = "CLAMPED (ATTACK QUARANTINE)" if (composite_score > current_threshold) else "ADAPTING TO LOAD"

    # Classification against active threshold
    is_anomaly = bool(composite_score > current_threshold)
    severity = "CRITICAL" if composite_score > current_threshold * 1.35 or effective_status >= 500 else ("WARNING" if is_anomaly else "NORMAL")

    # Threat Classification & Actionable Recommendations
    if effective_status >= 500:
        threat_type = f"Target Service Down / HTTP {effective_status}"
        action = f"Check {clean_url} upstream server health, container status & reverse proxy"
    elif effective_status == 429:
        threat_type = "Rate Limit Exhaustion / Layer 7 Throttling"
        action = "Back off request frequency or increase API token rate limits"
    elif effective_latency_ms > base_median_latency * 3.0:
        threat_type = "Volumetric Latency Surge / Suspected Congestion"
        action = f"Inspect edge routing and enable CDN cache shield on {clean_url}"
    elif jitter_ms > 220.0:
        threat_type = "Severe Packet Jitter / Route Flapping"
        action = "Audit BGP route health and intermediate hop packet loss"
    elif is_anomaly:
        threat_type = "Unusual Network Telemetry Deviation"
        action = "Monitor subsequent flows; quarantine anomalous session if persistent"
    else:
        threat_type = "Normal Service Telemetry"
        action = "No action required (nominal traffic)"

    flow_count = len(target_history) + 1
    flow_id_str = f"PROBE-{flow_count:05d}"
    now_str = time.strftime("%H:%M:%S")

    probe_record = {
        "flow_id": flow_count,
        "flow_id_str": flow_id_str,
        "timestamp": now_str,
        "target_url": clean_url,
        "status_code": effective_status,
        "latency_ms": round(effective_latency_ms, 1),
        "baseline_latency_ms": round(base_median_latency, 1),
        "jitter_ms": round(jitter_ms, 1),
        "content_length": content_length,
        "score": composite_score,
        "threshold": current_threshold,
        "threshold_stabilized": tau_stabilized,
        "threshold_rolling_mad": tau_rolling_mad,
        "threshold_fixed": tau_fixed,
        "threshold_drift": round(current_threshold - tau_0, 2),
        "adaptation_status": drift_status,
        "adaptation_mode": adaptation_mode,
        "is_anomaly": is_anomaly,
        "severity": severity,
        "threat_type": threat_type,
        "action": action,
        "error": error_msg,
        "simulated": bool(sim_mode),
    }

    # Append to rolling history
    target_history.append(probe_record)
    if len(target_history) > 200:
        target_history.pop(0)

    # Compute aggregate stats
    total_probes = len(target_history)
    anom_count = sum(1 for p in target_history if p["is_anomaly"])
    crit_count = sum(1 for p in target_history if p["severity"] == "CRITICAL")
    avg_latency = float(np.mean([p["latency_ms"] for p in target_history]))

    return {
        "status": "success",
        "probe": probe_record,
        "metrics": {
            "total_probes": total_probes,
            "anomalies_detected": anom_count,
            "critical_count": crit_count,
            "anomaly_rate_pct": round((anom_count / max(1, total_probes)) * 100, 1),
            "avg_latency_ms": round(avg_latency, 1),
            "baseline_median_ms": round(base_median_latency, 1),
            "current_threshold": current_threshold,
        },
        "recent_window": target_history[-40:],
        "active_simulation": sim_mode,
    }


@router.post("/set-simulation")
async def set_attack_simulation(url: str = Query(...), mode: Optional[str] = Query(None)):
    """Toggles attack simulation on a target (e.g. 'ddos_spike', 'gateway_error', or None to clear)."""
    clean_url = normalize_target_url(url)
    target_key = clean_url.lower()

    if mode in ["ddos_spike", "gateway_error", "latency_degradation", "jitter_burst"]:
        attack_simulation_state[target_key] = mode
    else:
        attack_simulation_state[target_key] = None

    return {"status": "success", "target": clean_url, "simulation_mode": attack_simulation_state[target_key]}


@router.post("/clear-history")
async def clear_probe_history(url: Optional[str] = Query(None)):
    """Clears probe telemetry history."""
    if url:
        clean_url = normalize_target_url(url).lower()
        history_store.pop(clean_url, None)
        attack_simulation_state.pop(clean_url, None)
    else:
        history_store.clear()
        attack_simulation_state.clear()
    return {"status": "success", "cleared": True}
