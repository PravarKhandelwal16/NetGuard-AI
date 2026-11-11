import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import {
  ShieldAlert,
  ShieldCheck,
  Activity,
  Play,
  Pause,
  RotateCcw,
  Volume2,
  VolumeX,
  AlertTriangle,
  Zap,
  TrendingUp,
  Cpu,
  Layers,
  FileText,
  Sliders,
  CheckCircle2,
  XCircle,
  ExternalLink,
  Download,
  Search,
  Filter,
  RefreshCw,
  Globe,
  Radio,
  Server,
  ArrowRight,
  Sparkles,
} from 'lucide-react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ReferenceLine,
} from 'recharts';
import { api } from '../services/api';

// Web Audio API Synth Alert Chime
function playAlertChime() {
  try {
    const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
    if (!AudioContextClass) return;
    const ctx = new AudioContextClass();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = 'triangle';
    osc.frequency.setValueAtTime(880, ctx.currentTime);
    osc.frequency.exponentialRampToValueAtTime(440, ctx.currentTime + 0.22);
    gain.gain.setValueAtTime(0.18, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.22);
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start();
    osc.stop(ctx.currentTime + 0.22);
  } catch (err) {
    // Ignore audio permission errors
  }
}

interface LiveProbeItem {
  flow_id: number;
  flow_id_str: string;
  timestamp: string;
  target_url: string;
  status_code: number;
  latency_ms: number;
  baseline_latency_ms: number;
  jitter_ms: number;
  content_length: number;
  score: number;
  threshold: number;
  threshold_stabilized?: number;
  threshold_rolling_mad?: number;
  threshold_fixed?: number;
  threshold_drift?: number;
  adaptation_status?: string;
  adaptation_mode?: string;
  is_anomaly: boolean;
  severity: 'NORMAL' | 'WARNING' | 'CRITICAL';
  threat_type: string;
  action: string;
  error?: string | null;
  simulated: boolean;
}

interface AlertItem {
  id: string;
  flow_id_str: string;
  timestamp: string;
  severity: 'CRITICAL' | 'WARNING';
  score: number;
  threshold: number;
  latency_ms: number;
  status_code: number;
  threat_type: string;
  action: string;
  target_url: string;
}

export const SinglePageDashboard: React.FC = () => {
  // Target Service Configuration
  const [targetUrl, setTargetUrl] = useState('https://google.com');
  const [selectedTarget, setSelectedTarget] = useState('https://google.com');
  const [pollIntervalMs, setPollIntervalMs] = useState(1500);
  const [audioEnabled, setAudioEnabled] = useState(true);

  // Stream State — DOES NOT START DIRECTLY ON LOAD
  const [isMonitoring, setIsMonitoring] = useState(false);
  const [isProbingInFlight, setIsProbingInFlight] = useState(false);

  // Attack Simulation State
  const [activeSimulation, setActiveSimulation] = useState<string | null>(null);

  // AI & Detection Tuning Parameters
  const [alpha, setAlpha] = useState(0.5);
  const [baseThreshold, setBaseThreshold] = useState(32.77);
  const [engineMode, setEngineMode] = useState<'stabilized' | 'rolling_mad' | 'fixed'>('stabilized');
  const [compareRunaway, setCompareRunaway] = useState(true);

  // Telemetry & Alerts State
  const [history, setHistory] = useState<LiveProbeItem[]>([]);
  const [alerts, setAlerts] = useState<AlertItem[]>([]);
  const [latestProbe, setLatestProbe] = useState<LiveProbeItem | null>(null);
  const [metrics, setMetrics] = useState({
    total_probes: 0,
    anomalies_detected: 0,
    critical_count: 0,
    anomaly_rate_pct: 0,
    avg_latency_ms: 0,
    baseline_median_ms: 0,
    current_threshold: 32.77,
  });

  // Table filters & search
  const [tableFilter, setTableFilter] = useState<'ALL' | 'ANOMALIES' | 'CRITICAL'>('ALL');
  const [tableSearch, setTableSearch] = useState('');

  // Probing Ref to avoid stale state in async timeout loop
  const isMonitoringRef = useRef(isMonitoring);
  isMonitoringRef.current = isMonitoring;

  const activeSimulationRef = useRef(activeSimulation);
  activeSimulationRef.current = activeSimulation;

  const selectedTargetRef = useRef(selectedTarget);
  selectedTargetRef.current = selectedTarget;

  const alphaRef = useRef(alpha);
  alphaRef.current = alpha;

  const baseThresholdRef = useRef(baseThreshold);
  baseThresholdRef.current = baseThreshold;

  const engineModeRef = useRef(engineMode);
  engineModeRef.current = engineMode;

  const pollIntervalRef = useRef(pollIntervalMs);
  pollIntervalRef.current = pollIntervalMs;

  const inFlightRef = useRef(false);
  const [streamError, setStreamError] = useState<string | null>(null);

  // Single probe execution without state-closure lock
  const doProbe = async () => {
    if (inFlightRef.current) return;
    inFlightRef.current = true;
    setIsProbingInFlight(true);
    try {
      const res = await api.probeLiveService({
        url: selectedTargetRef.current,
        simulate_attack: activeSimulationRef.current,
        alpha: alphaRef.current,
        base_threshold: baseThresholdRef.current,
        engine_mode: engineModeRef.current,
      });

      const p = res.probe;
      setLatestProbe(p);
      setMetrics(res.metrics);
      setStreamError(null);

      setHistory((prev) => [p, ...prev.slice(0, 99)]);

      if (p.is_anomaly) {
        const newAlert: AlertItem = {
          id: `${p.flow_id}-${Date.now()}`,
          flow_id_str: p.flow_id_str,
          timestamp: p.timestamp,
          severity: p.severity === 'CRITICAL' ? 'CRITICAL' : 'WARNING',
          score: p.score,
          threshold: p.threshold,
          latency_ms: p.latency_ms,
          status_code: p.status_code,
          threat_type: p.threat_type,
          action: p.action,
          target_url: p.target_url,
        };

        setAlerts((prev) => [newAlert, ...prev.slice(0, 49)]);

        if (audioEnabled && p.severity === 'CRITICAL') {
          playAlertChime();
        }
      }
    } catch (err: any) {
      console.warn('Probe tick error:', err);
      setStreamError(err?.message || 'Connection error');
    } finally {
      inFlightRef.current = false;
      setIsProbingInFlight(false);
    }
  };

  // Robust, self-sustaining loop that never deadlocks
  useEffect(() => {
    if (!isMonitoring) return;

    let isCancelled = false;
    let timerId: any = null;

    const probeLoop = async () => {
      if (isCancelled) return;
      await doProbe();
      if (!isCancelled && isMonitoringRef.current) {
        timerId = setTimeout(probeLoop, pollIntervalRef.current);
      }
    };

    probeLoop();

    return () => {
      isCancelled = true;
      if (timerId) clearTimeout(timerId);
    };
  }, [isMonitoring]);

  // Derived Streaming Window Data for Chart
  const chartData = useMemo(() => {
    return [...history.slice(0, 45)].reverse().map((item) => ({
      step: item.flow_id_str,
      score: item.score,
      threshold: item.threshold,
      threshold_stabilized: item.threshold_stabilized ?? item.threshold,
      threshold_rolling_mad: item.threshold_rolling_mad ?? item.threshold,
      threshold_fixed: item.threshold_fixed ?? baseThreshold,
      adaptation_status: item.adaptation_status,
      latency: item.latency_ms,
      status: item.status_code,
      is_anomaly: item.is_anomaly,
      severity: item.severity,
      threat: item.threat_type,
    }));
  }, [history, baseThreshold]);

  // System Overall Threat Level
  const threatLevel = useMemo(() => {
    if (!isMonitoring && history.length === 0) return 'IDLE';
    if (history.length === 0) return 'HEALTHY';
    const recent = history.slice(0, 10);
    const recentAnoms = recent.filter((p) => p.is_anomaly).length;
    const hasCrit = recent.some((p) => p.severity === 'CRITICAL');
    if (hasCrit || recentAnoms >= 3) return 'CRITICAL';
    if (recentAnoms >= 1) return 'ELEVATED';
    return 'HEALTHY';
  }, [history, isMonitoring]);

  // Filtered Table Items
  const filteredHistory = useMemo(() => {
    return history.filter((item) => {
      if (tableFilter === 'ANOMALIES' && !item.is_anomaly) return false;
      if (tableFilter === 'CRITICAL' && item.severity !== 'CRITICAL') return false;

      if (tableSearch.trim()) {
        const q = tableSearch.toLowerCase();
        const idMatch = item.flow_id_str.toLowerCase().includes(q);
        const urlMatch = item.target_url.toLowerCase().includes(q);
        const threatMatch = item.threat_type.toLowerCase().includes(q);
        const statusMatch = String(item.status_code).includes(q);
        return idMatch || urlMatch || threatMatch || statusMatch;
      }
      return true;
    });
  }, [history, tableFilter, tableSearch]);

  const handleStartStream = () => {
    if (targetUrl.trim()) {
      setSelectedTarget(targetUrl.trim());
      setIsMonitoring(true);
    }
  };

  const handlePresetSelect = (url: string) => {
    setTargetUrl(url);
    setSelectedTarget(url);
    // Don't auto-start: let user review selection and click "Start Reading & Detecting"
  };

  const handleResetHistory = () => {
    setHistory([]);
    setAlerts([]);
    setLatestProbe(null);
    setMetrics({
      total_probes: 0,
      anomalies_detected: 0,
      critical_count: 0,
      anomaly_rate_pct: 0,
      avg_latency_ms: 0,
      baseline_median_ms: 0,
      current_threshold: baseThreshold,
    });
  };

  const handleExportCSV = () => {
    if (history.length === 0) return;
    const headers = [
      'Flow ID',
      'Timestamp',
      'Target URL',
      'Status Code',
      'Latency (ms)',
      'Baseline Latency (ms)',
      'Jitter (ms)',
      'Anomaly Score',
      'Threshold',
      'Is Anomaly',
      'Severity',
      'Threat Classification',
      'Remediation Action',
    ];
    const rows = history.map((h) => [
      h.flow_id_str,
      h.timestamp,
      h.target_url,
      h.status_code,
      h.latency_ms,
      h.baseline_latency_ms,
      h.jitter_ms,
      h.score,
      h.threshold,
      h.is_anomaly ? 'TRUE' : 'FALSE',
      h.severity,
      `"${h.threat_type}"`,
      `"${h.action}"`,
    ]);
    const csvContent =
      'data:text/csv;charset=utf-8,' +
      [headers.join(','), ...rows.map((e) => e.join(','))].join('\n');
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', `netguard_live_telemetry_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="min-h-screen bg-[#F5F3EC] text-stone-900 flex flex-col font-sans selection:bg-rose-200">
      {/* Top Application Header */}
      <header className="bg-[#131316] text-stone-100 px-6 py-4 flex flex-wrap items-center justify-between gap-4 border-b border-stone-800 shadow-md">
        <div className="flex items-center space-x-3">
          <div className="w-9 h-9 rounded-lg bg-rose-500/20 border border-rose-500/40 flex items-center justify-center text-rose-400">
            <Radio className={`w-5 h-5 ${isMonitoring ? 'animate-pulse text-emerald-400' : 'text-stone-400'}`} />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <h1 className="text-lg font-bold tracking-tight text-white">NETGUARD AI</h1>
              <span
                className={`text-[10px] uppercase font-semibold px-2 py-0.5 rounded border flex items-center space-x-1 ${
                  isMonitoring
                    ? 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30'
                    : 'bg-stone-800 text-stone-400 border-stone-700'
                }`}
              >
                {isMonitoring && <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-ping" />}
                <span>{isMonitoring ? 'Live Streaming' : 'Standby / Idle'}</span>
              </span>
            </div>
            <p className="text-xs text-stone-400">
              Live Network Endpoint Telemetry &bull; Self-Tuning Threshold Anomaly Detection &bull; Real-Time Alerts
            </p>
          </div>
        </div>

        {/* Master Stream Controls */}
        <div className="flex items-center space-x-3 bg-stone-900/80 px-4 py-2 rounded-xl border border-stone-800">
          <button
            onClick={() => {
              if (isMonitoring) {
                setIsMonitoring(false);
              } else {
                handleStartStream();
              }
            }}
            className={`px-4 py-2 rounded-lg font-bold text-xs flex items-center space-x-2 transition-all ${
              isMonitoring
                ? 'bg-amber-500 text-stone-950 hover:bg-amber-400 shadow-lg shadow-amber-500/20'
                : 'bg-emerald-500 text-stone-950 hover:bg-emerald-400 shadow-lg shadow-emerald-500/30 ring-2 ring-emerald-400/40'
            }`}
          >
            {isMonitoring ? (
              <>
                <Pause className="w-4 h-4 fill-current" />
                <span>Pause Detection</span>
              </>
            ) : (
              <>
                <Play className="w-4 h-4 fill-current" />
                <span>Start Reading & Detecting</span>
              </>
            )}
          </button>

          <button
            onClick={handleResetHistory}
            title="Reset Buffer & Clears Metrics"
            className="p-2 rounded-lg bg-stone-800 hover:bg-stone-700 text-stone-300 transition-colors"
          >
            <RotateCcw className="w-4 h-4" />
          </button>

          <div className="h-5 w-[1px] bg-stone-700" />

          {/* Stream Rate Selector */}
          <div className="flex items-center space-x-1 text-xs">
            <span className="text-stone-400 mr-1 text-[11px]">Interval:</span>
            {[
              { label: '500ms', ms: 500 },
              { label: '1s', ms: 1000 },
              { label: '2s', ms: 2000 },
            ].map((spd) => (
              <button
                key={spd.label}
                onClick={() => setPollIntervalMs(spd.ms)}
                className={`px-2 py-1 rounded text-[11px] font-mono transition-colors ${
                  pollIntervalMs === spd.ms
                    ? 'bg-stone-200 text-stone-900 font-bold'
                    : 'text-stone-400 hover:text-stone-200 hover:bg-stone-800'
                }`}
              >
                {spd.label}
              </button>
            ))}
          </div>

          <div className="h-5 w-[1px] bg-stone-700" />

          {/* Audio Chime Toggle */}
          <button
            onClick={() => setAudioEnabled(!audioEnabled)}
            title={audioEnabled ? 'Audio alerts active' : 'Audio alerts muted'}
            className={`p-2 rounded-lg transition-colors ${
              audioEnabled ? 'bg-rose-500/20 text-rose-300' : 'bg-stone-800 text-stone-500'
            }`}
          >
            {audioEnabled ? <Volume2 className="w-4 h-4" /> : <VolumeX className="w-4 h-4" />}
          </button>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 md:p-6 space-y-6">
        {/* Step 1: Select Website / Service & Options (Does Not Auto-Start) */}
        <section className="bg-white rounded-2xl border border-stone-200 p-5 shadow-sm space-y-4">
          <div className="flex flex-col space-y-1">
            <div className="flex items-center space-x-2">
              <span className="w-5 h-5 rounded-full bg-stone-900 text-white text-[11px] font-bold flex items-center justify-center">
                1
              </span>
              <h2 className="text-sm font-bold text-stone-900 uppercase tracking-wider">
                Choose Target Website or Service
              </h2>
            </div>
            <p className="text-xs text-stone-500 pl-7">
              Select a preset or enter any live website or API endpoint to monitor. Click &ldquo;Start Reading &amp; Detecting&rdquo; when ready.
            </p>
          </div>

          <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
            {/* Live Service Input Field */}
            <div className="flex-1 flex items-center gap-2">
              <div className="relative flex-1">
                <Globe className="w-4 h-4 text-stone-400 absolute left-3 top-3" />
                <input
                  type="text"
                  value={targetUrl}
                  onChange={(e) => setTargetUrl(e.target.value)}
                  placeholder="Enter website URL or service endpoint (e.g. https://google.com)"
                  className="w-full pl-9 pr-4 py-2 bg-stone-50 border border-stone-200 rounded-xl text-xs sm:text-sm font-mono text-stone-800 placeholder-stone-400 focus:outline-none focus:ring-2 focus:ring-stone-400"
                />
              </div>

              {!isMonitoring ? (
                <button
                  type="button"
                  onClick={handleStartStream}
                  className="px-5 py-2.5 bg-emerald-600 hover:bg-emerald-500 text-white font-bold text-xs rounded-xl shadow-md hover:shadow-emerald-500/20 transition-all flex items-center space-x-2 whitespace-nowrap"
                >
                  <Play className="w-4 h-4 fill-current" />
                  <span>Start Reading &amp; Detecting</span>
                </button>
              ) : (
                <button
                  type="button"
                  onClick={() => setIsMonitoring(false)}
                  className="px-5 py-2.5 bg-amber-500 hover:bg-amber-400 text-stone-950 font-bold text-xs rounded-xl shadow-md transition-all flex items-center space-x-2 whitespace-nowrap"
                >
                  <Pause className="w-4 h-4 fill-current" />
                  <span>Pause Stream</span>
                </button>
              )}
            </div>

            {/* Quick-Connect Presets */}
            <div className="flex flex-wrap items-center gap-1.5">
              <span className="text-[11px] font-bold text-stone-400 uppercase tracking-wider mr-1">
                Presets:
              </span>
              {[
                { label: 'Google', url: 'https://google.com' },
                { label: 'Cloudflare', url: 'https://cloudflare.com' },
                { label: 'GitHub', url: 'https://github.com' },
                { label: 'Local API', url: 'http://localhost:8000/api/health' },
              ].map((p) => (
                <button
                  key={p.label}
                  onClick={() => handlePresetSelect(p.url)}
                  className={`px-2.5 py-1.5 rounded-lg text-xs font-medium border transition-all ${
                    targetUrl === p.url
                      ? 'bg-stone-900 text-white border-stone-900 shadow-xs font-bold'
                      : 'bg-stone-50 text-stone-600 border-stone-200 hover:bg-stone-100'
                  }`}
                >
                  {p.label}
                </button>
              ))}
            </div>
          </div>

          {/* Interactive AI Attack Simulation Controls */}
          <div className="pt-3 border-t border-stone-100 flex flex-wrap items-center justify-between gap-3 text-xs">
            <div className="flex flex-wrap items-center gap-2">
              <span className="font-semibold text-stone-700 flex items-center space-x-1">
                <Zap className="w-3.5 h-3.5 text-amber-500" />
                <span>Simulate Attack &amp; Test Detection:</span>
              </span>

              {[
                { id: 'ddos_spike', label: '💥 Volumetric DDoS Burst', color: 'hover:bg-rose-50 text-rose-700 border-rose-200' },
                { id: 'gateway_error', label: '⚠️ 502 Bad Gateway Outage', color: 'hover:bg-amber-50 text-amber-700 border-amber-200' },
                { id: 'jitter_burst', label: '⚡ Packet Loss & Jitter', color: 'hover:bg-purple-50 text-purple-700 border-purple-200' },
              ].map((sim) => (
                <button
                  key={sim.id}
                  onClick={() => setActiveSimulation(activeSimulation === sim.id ? null : sim.id)}
                  className={`px-3 py-1 rounded-lg border font-medium transition-all ${
                    activeSimulation === sim.id
                      ? 'bg-rose-600 text-white border-rose-600 shadow-sm ring-2 ring-rose-400/40'
                      : `bg-white ${sim.color}`
                  }`}
                >
                  {sim.label}
                </button>
              ))}

              {activeSimulation && (
                <button
                  onClick={() => setActiveSimulation(null)}
                  className="px-2 py-1 text-stone-500 hover:text-stone-800 underline text-[11px]"
                >
                  Reset to Normal Traffic
                </button>
              )}
            </div>

            {/* Target Live Status Indicator */}
            <div className="flex items-center space-x-2 text-stone-600">
              <span
                className={`w-2 h-2 rounded-full ${
                  isMonitoring ? 'bg-emerald-500 animate-ping' : 'bg-stone-400'
                }`}
              />
              <span>
                Selected: <strong className="font-mono text-stone-800">{selectedTarget}</strong>
              </span>
              <span>&bull;</span>
              <span>
                Status:{' '}
                <strong className={isMonitoring ? (streamError ? 'text-rose-600' : 'text-emerald-700') : 'text-stone-500'}>
                  {isMonitoring ? (streamError ? `Retrying (${streamError})` : (isProbingInFlight ? 'Probing...' : 'Streaming Active')) : 'Waiting for Start'}
                </strong>
              </span>
              <span>&bull;</span>
              <span>Pings: {metrics.total_probes}</span>
            </div>
          </div>
        </section>

        {/* 4 Pastel Hero KPI Cards */}
        <section className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {/* Card 1: System Threat Status */}
          <div
            className={`p-5 rounded-2xl border transition-all ${
              threatLevel === 'CRITICAL'
                ? 'bg-[#FCE7F3] border-rose-300 shadow-md ring-2 ring-rose-400/30'
                : threatLevel === 'ELEVATED'
                ? 'bg-[#FEF3C7] border-amber-300 shadow-sm'
                : threatLevel === 'IDLE'
                ? 'bg-stone-100 border-stone-200 text-stone-500'
                : 'bg-[#DCFCE7] border-emerald-300 shadow-sm'
            }`}
          >
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-bold uppercase tracking-wider text-stone-700">
                Threat Status
              </span>
              {threatLevel === 'CRITICAL' ? (
                <ShieldAlert className="w-5 h-5 text-rose-600 animate-bounce" />
              ) : threatLevel === 'ELEVATED' ? (
                <AlertTriangle className="w-5 h-5 text-amber-600" />
              ) : threatLevel === 'IDLE' ? (
                <Activity className="w-5 h-5 text-stone-400" />
              ) : (
                <ShieldCheck className="w-5 h-5 text-emerald-700" />
              )}
            </div>
            <div className="text-2xl font-black tracking-tight text-stone-900">
              {threatLevel === 'CRITICAL'
                ? 'CRITICAL ATTACK'
                : threatLevel === 'ELEVATED'
                ? 'ELEVATED RISK'
                : threatLevel === 'IDLE'
                ? 'STANDBY / IDLE'
                : 'NORMAL / NOMINAL'}
            </div>
            <p className="text-xs text-stone-600 mt-1">
              {threatLevel === 'CRITICAL'
                ? 'Security & latency anomalies detected'
                : threatLevel === 'ELEVATED'
                ? 'Variance above baseline threshold'
                : threatLevel === 'IDLE'
                ? 'Click "Start Reading & Detecting" to begin'
                : 'Healthy continuous telemetry stream'}
            </p>
          </div>

          {/* Card 2: Current Score vs Self-Tuning Threshold */}
          <div className="bg-[#FEF3C7] p-5 rounded-2xl border border-amber-200/80 shadow-sm flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-bold uppercase tracking-wider text-stone-700">
                  Live Anomaly Score vs &tau;
                </span>
                <Activity className="w-5 h-5 text-amber-700" />
              </div>
              <div className="flex items-baseline space-x-2">
                <span className="text-2xl font-black tracking-tight text-stone-900">
                  {latestProbe ? latestProbe.score.toFixed(2) : '0.00'}
                </span>
                <span className="text-xs font-mono text-stone-700 font-bold">
                  / &tau;: {latestProbe ? latestProbe.threshold.toFixed(2) : baseThreshold.toFixed(2)}
                </span>
              </div>
              <div className="w-full bg-amber-200/60 rounded-full h-1.5 mt-2.5 overflow-hidden">
                <div
                  className={`h-full transition-all duration-300 ${
                    latestProbe && latestProbe.score > latestProbe.threshold
                      ? 'bg-rose-600'
                      : 'bg-amber-600'
                  }`}
                  style={{
                    width: `${Math.min(
                      100,
                      ((latestProbe?.score || 0) / (latestProbe?.threshold || 32.77)) * 50
                    )}%`,
                  }}
                />
              </div>
            </div>

            {/* Threshold Adaptation State Indicator */}
            <div className="pt-2.5 mt-2 border-t border-amber-200/60 flex items-center justify-between text-[11px]">
              <span className="text-stone-600 font-medium">Adaptation Status:</span>
              <span
                className={`px-2 py-0.5 rounded-md font-bold text-[10px] tracking-tight ${
                  latestProbe?.adaptation_status?.includes('RUNAWAY')
                    ? 'bg-rose-600 text-white animate-pulse'
                    : latestProbe?.adaptation_status?.includes('QUARANTINE') || latestProbe?.adaptation_status?.includes('CLAMPED')
                    ? 'bg-purple-100 text-purple-800 border border-purple-300'
                    : latestProbe?.adaptation_status?.includes('ADAPTING')
                    ? 'bg-emerald-100 text-emerald-800 border border-emerald-300'
                    : 'bg-amber-100 text-amber-800 border border-amber-300'
                }`}
              >
                {latestProbe?.adaptation_status || 'ADAPTING TO LOAD'}
              </span>
            </div>
          </div>

          {/* Card 3: Network Latency & Jitter */}
          <div className="bg-[#DBEAFE] p-5 rounded-2xl border border-blue-200/80 shadow-sm">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-bold uppercase tracking-wider text-stone-700">
                Live Ping &amp; Jitter
              </span>
              <Server className="w-5 h-5 text-blue-700" />
            </div>
            <div className="text-2xl font-black tracking-tight text-stone-900">
              {latestProbe ? `${latestProbe.latency_ms} ms` : '-- ms'}
            </div>
            <div className="flex items-center justify-between text-xs text-stone-600 mt-1">
              <span>Baseline: {metrics.baseline_median_ms ? `${metrics.baseline_median_ms} ms` : '--'}</span>
              <span className="font-mono">
                {latestProbe ? `Jitter: ±${latestProbe.jitter_ms} ms` : 'Standby'}
              </span>
            </div>
          </div>

          {/* Card 4: Anomalies Intercepted */}
          <div className="bg-[#FCE7F3] p-5 rounded-2xl border border-rose-200/80 shadow-sm">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-bold uppercase tracking-wider text-stone-700">
                AI Alerts Intercepted
              </span>
              <ShieldAlert className="w-5 h-5 text-rose-600" />
            </div>
            <div className="text-2xl font-black tracking-tight text-rose-950">
              {metrics.anomalies_detected}
              <span className="text-sm font-semibold text-rose-700 ml-2">
                ({metrics.anomaly_rate_pct}%)
              </span>
            </div>
            <p className="text-xs text-stone-600 mt-1">
              Critical Outages / Spikes: <span className="font-bold">{metrics.critical_count}</span>
            </p>
          </div>
        </section>

        {/* Real-Time Telemetry & Anomaly Score Chart */}
        <section className="bg-white rounded-2xl border border-stone-200 p-6 shadow-sm space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-2 border-b border-stone-100">
            <div>
              <h2 className="text-lg font-bold text-stone-900 flex items-center space-x-2">
                <span>Real-Time Anomaly Score vs Self-Tuning Threshold</span>
              </h2>
              <p className="text-xs text-stone-500">
                Multi-signal telemetry fusion (S_t = &alpha; &middot; latency_delta + (1-&alpha;) &middot; jitter_drift). Live threshold &tau;_t adapts to legitimate traffic drift.
              </p>
            </div>

            {/* Threshold Method & Sensitivity Slider */}
            <div className="flex flex-wrap items-center gap-3 text-xs">
              <div className="flex items-center space-x-2">
                <span className="text-stone-500">Sensitivity &alpha;:</span>
                <input
                  type="range"
                  min="0.1"
                  max="0.9"
                  step="0.1"
                  value={alpha}
                  onChange={(e) => setAlpha(parseFloat(e.target.value))}
                  className="w-20 accent-rose-500 cursor-pointer"
                />
                <span className="font-mono font-bold text-stone-700">{alpha.toFixed(1)}</span>
              </div>

              {/* Dynamic Threshold Engine Selector */}
              <div className="flex items-center space-x-1 bg-stone-100 p-0.5 rounded-xl border border-stone-200">
                <button
                  type="button"
                  onClick={() => setEngineMode('stabilized')}
                  className={`px-2.5 py-1 rounded-lg text-xs font-semibold transition-all ${
                    engineMode === 'stabilized'
                      ? 'bg-amber-500 text-stone-950 shadow-xs'
                      : 'text-stone-600 hover:text-stone-900'
                  }`}
                  title="Bounded Clamping + Attack Quarantine + Smoothing (Prevents Runaway)"
                >
                  Stabilized Adaptive
                </button>
                <button
                  type="button"
                  onClick={() => setEngineMode('rolling_mad')}
                  className={`px-2.5 py-1 rounded-lg text-xs font-semibold transition-all ${
                    engineMode === 'rolling_mad'
                      ? 'bg-purple-600 text-white shadow-xs'
                      : 'text-stone-600 hover:text-stone-900'
                  }`}
                  title="Naive Rolling MAD: Suffers Threshold Runaway during sustained attacks"
                >
                  Rolling MAD
                </button>
                <button
                  type="button"
                  onClick={() => setEngineMode('fixed')}
                  className={`px-2.5 py-1 rounded-lg text-xs font-semibold transition-all ${
                    engineMode === 'fixed'
                      ? 'bg-stone-800 text-white shadow-xs'
                      : 'text-stone-600 hover:text-stone-900'
                  }`}
                  title="Rigid Static Threshold: Cannot adapt to natural network variance"
                >
                  Fixed Baseline
                </button>
              </div>

              {/* Runaway Comparison Toggle */}
              {engineMode === 'stabilized' && (
                <label className="flex items-center space-x-1.5 cursor-pointer text-stone-600 select-none bg-stone-50 border border-stone-200 px-2 py-1 rounded-lg hover:bg-stone-100">
                  <input
                    type="checkbox"
                    checked={compareRunaway}
                    onChange={(e) => setCompareRunaway(e.target.checked)}
                    className="accent-purple-600 rounded cursor-pointer"
                  />
                  <span className="font-medium text-[11px]">Compare Runaway (Rolling MAD)</span>
                </label>
              )}
            </div>
          </div>

          {/* Recharts Live Line Graph */}
          <div className="h-72 w-full pt-2">
            {chartData.length === 0 ? (
              <div className="h-full flex flex-col items-center justify-center text-center p-6 text-stone-400 bg-stone-50/50 rounded-xl border border-dashed border-stone-200">
                <Activity className="w-8 h-8 text-stone-300 mb-2" />
                <p className="text-sm font-semibold text-stone-700">Telemetry Stream Inactive</p>
                <p className="text-xs text-stone-500 mt-1 max-w-sm">
                  Choose a target website above and click &ldquo;Start Reading &amp; Detecting&rdquo; to plot live anomaly scores and dynamic thresholds.
                </p>
              </div>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
                  <XAxis
                    dataKey="step"
                    tick={{ fontSize: 10, fill: '#78716C' }}
                    tickLine={false}
                    axisLine={{ stroke: '#E7E5E4' }}
                  />
                  <YAxis
                    domain={[0, 'auto']}
                    tick={{ fontSize: 10, fill: '#78716C' }}
                    tickLine={false}
                    axisLine={{ stroke: '#E7E5E4' }}
                  />
                  <Tooltip
                    content={({ active, payload }) => {
                      if (active && payload && payload.length) {
                        const data = payload[0].payload;
                        return (
                          <div className="bg-[#1C1917] text-white p-3 rounded-xl shadow-xl text-xs border border-stone-700 space-y-1">
                            <div className="font-bold flex items-center justify-between space-x-3">
                              <span>{data.step}</span>
                              <span
                                className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${
                                  data.is_anomaly
                                    ? 'bg-rose-500 text-white'
                                    : 'bg-emerald-500 text-stone-950'
                                }`}
                              >
                                {data.is_anomaly ? 'ANOMALY DETECTED' : 'NOMINAL'}
                              </span>
                            </div>
                            <div>Score: <span className="font-mono text-amber-300 font-bold">{data.score.toFixed(2)}</span></div>
                            <div>Active Threshold: <span className="font-mono text-stone-300 font-bold">{data.threshold.toFixed(2)}</span></div>
                            {compareRunaway && data.threshold_rolling_mad !== undefined && (
                              <div>Rolling MAD τ: <span className="font-mono text-purple-300">{data.threshold_rolling_mad.toFixed(2)}</span></div>
                            )}
                            {data.adaptation_status && (
                              <div className="text-[10px] text-emerald-400 font-semibold">{data.adaptation_status}</div>
                            )}
                            <div>Latency: <span className="font-mono text-blue-300 font-bold">{data.latency} ms</span></div>
                            <div>Status: <span className="font-mono text-stone-200 font-bold">{data.status}</span></div>
                            <div className="text-[11px] text-stone-400 pt-1 border-t border-stone-700">
                              {data.threat}
                            </div>
                          </div>
                        );
                      }
                      return null;
                    }}
                  />

                  {/* Active Adaptive Threshold Line */}
                  <Line
                    type="monotone"
                    dataKey="threshold"
                    stroke="#F59E0B"
                    strokeWidth={2}
                    strokeDasharray="4 4"
                    dot={false}
                    isAnimationActive={false}
                    name="Active Threshold (τ)"
                  />

                  {/* Runaway Rolling MAD Reference Line (Demonstrating Runaway vs Stabilized) */}
                  {compareRunaway && engineMode === 'stabilized' && (
                    <Line
                      type="monotone"
                      dataKey="threshold_rolling_mad"
                      stroke="#A855F7"
                      strokeWidth={1.8}
                      strokeDasharray="2 2"
                      dot={false}
                      isAnimationActive={false}
                      name="Rolling MAD (Runaway Risk)"
                    />
                  )}

                  {/* Composite Anomaly Score Line */}
                  <Line
                    type="monotone"
                    dataKey="score"
                    stroke="#3B82F6"
                    strokeWidth={2.5}
                    dot={(props: any) => {
                      const { cx, cy, payload } = props;
                      if (payload && payload.is_anomaly) {
                        return (
                          <circle
                            key={`dot-${cx}-${cy}`}
                            cx={cx}
                            cy={cy}
                            r={5.5}
                            fill="#EF4444"
                            stroke="#FFFFFF"
                            strokeWidth={2}
                            className="animate-pulse"
                          />
                        );
                      }
                      return null;
                    }}
                    isAnimationActive={false}
                    name="Anomaly Score (S)"
                  />
                </LineChart>
              </ResponsiveContainer>
            )}
          </div>

          <div className="flex flex-wrap items-center justify-between text-xs text-stone-500 pt-2 border-t border-stone-100 gap-2">
            <div className="flex flex-wrap items-center gap-4">
              <span className="flex items-center space-x-1.5">
                <span className="w-3 h-0.5 bg-blue-500" />
                <span>Multi-Signal Score (S_t)</span>
              </span>
              <span className="flex items-center space-x-1.5">
                <span className="w-3 h-0.5 bg-amber-500 border-dashed" />
                <span>Active Dynamic Threshold (&tau;_t)</span>
              </span>
              {compareRunaway && engineMode === 'stabilized' && (
                <span className="flex items-center space-x-1.5">
                  <span className="w-3 h-0.5 bg-purple-500 border-dotted" />
                  <span>Unbounded Rolling MAD (Demonstrates Runaway)</span>
                </span>
              )}
              <span className="flex items-center space-x-1.5">
                <span className="w-2.5 h-2.5 rounded-full bg-rose-500" />
                <span>Anomalous Burst / Attack Spike</span>
              </span>
            </div>
            <span>
              {isMonitoring ? 'Streaming live probes' : 'Standby mode (select options above)'}
            </span>
          </div>
        </section>

        {/* Live Incident Alerts & Streaming Table Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Live Incident Alerts Panel (Left Column) */}
          <section className="lg:col-span-1 bg-white rounded-2xl border border-stone-200 p-5 shadow-sm flex flex-col space-y-3">
            <div className="flex items-center justify-between pb-3 border-b border-stone-100">
              <div className="flex items-center space-x-2">
                <ShieldAlert className="w-5 h-5 text-rose-600" />
                <h3 className="text-sm font-bold text-stone-900">Live AI Security Alerts</h3>
              </div>
              <span className="text-xs font-bold px-2 py-0.5 rounded-full bg-rose-100 text-rose-800">
                {alerts.length} Incidents
              </span>
            </div>

            <div className="flex-1 overflow-y-auto max-h-[460px] space-y-2.5 pr-1">
              {alerts.length === 0 ? (
                <div className="h-64 flex flex-col items-center justify-center text-center p-6 text-stone-400">
                  <ShieldCheck className="w-10 h-10 text-emerald-400 mb-2" />
                  <p className="text-sm font-semibold text-stone-700">No Incidents Flagged</p>
                  <p className="text-xs mt-1">
                    {isMonitoring
                      ? 'Target telemetry is nominal. Use "Simulate Attack" above to test alert generation.'
                      : 'Stream paused. Click "Start Reading & Detecting" to begin surveillance.'}
                  </p>
                </div>
              ) : (
                alerts.map((al) => (
                  <div
                    key={al.id}
                    className={`p-3.5 rounded-xl border transition-all text-xs space-y-1.5 ${
                      al.severity === 'CRITICAL'
                        ? 'bg-rose-50/80 border-rose-200 shadow-sm'
                        : 'bg-amber-50/80 border-amber-200'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span
                        className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider ${
                          al.severity === 'CRITICAL'
                            ? 'bg-rose-500 text-white'
                            : 'bg-amber-500 text-stone-900'
                        }`}
                      >
                        {al.severity} &bull; HTTP {al.status_code}
                      </span>
                      <span className="text-[11px] font-mono text-stone-500">{al.timestamp}</span>
                    </div>

                    <div className="font-bold text-stone-900 pt-0.5">
                      {al.threat_type}
                    </div>

                    <div className="text-stone-600 text-[11px] flex items-center justify-between">
                      <span>Latency: <strong className="text-rose-700 font-mono">{al.latency_ms} ms</strong></span>
                      <span>
                        Score: <strong className="text-rose-600 font-mono">{al.score.toFixed(2)}</strong> (&tau;: {al.threshold.toFixed(2)})
                      </span>
                    </div>

                    <div className="pt-1 border-t border-stone-200/60 text-[11px] text-stone-700 flex items-start space-x-1.5">
                      <span className="font-semibold text-stone-900 whitespace-nowrap">Mitigation:</span>
                      <span className="text-stone-600">{al.action}</span>
                    </div>
                  </div>
                ))
              )}
            </div>

            {alerts.length > 0 && (
              <button
                onClick={() => setAlerts([])}
                className="w-full py-1.5 text-xs text-stone-500 hover:text-stone-800 border border-stone-200 rounded-lg hover:bg-stone-50 transition-colors"
              >
                Clear Alert Feed
              </button>
            )}
          </section>

          {/* Live Telemetry Flow Table (Right Columns) */}
          <section className="lg:col-span-2 bg-white rounded-2xl border border-stone-200 p-5 shadow-sm flex flex-col space-y-3">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-stone-100">
              <div>
                <h3 className="text-sm font-bold text-stone-900">Live Telemetry Inspector</h3>
                <p className="text-xs text-stone-500">Continuous network probe stream evaluated in real-time</p>
              </div>

              {/* Filter Pills and Search */}
              <div className="flex items-center space-x-2">
                <div className="relative">
                  <Search className="w-3.5 h-3.5 text-stone-400 absolute left-2.5 top-2.5" />
                  <input
                    type="text"
                    placeholder="Search URL, status, ID..."
                    value={tableSearch}
                    onChange={(e) => setTableSearch(e.target.value)}
                    className="pl-8 pr-3 py-1 bg-stone-50 border border-stone-200 rounded-lg text-xs text-stone-800 placeholder-stone-400 focus:outline-none focus:ring-1 focus:ring-stone-400"
                  />
                </div>

                <div className="flex rounded-lg bg-stone-100 p-0.5 text-[11px] font-medium">
                  {(['ALL', 'ANOMALIES', 'CRITICAL'] as const).map((mode) => (
                    <button
                      key={mode}
                      onClick={() => setTableFilter(mode)}
                      className={`px-2.5 py-1 rounded-md transition-colors ${
                        tableFilter === mode
                          ? 'bg-white text-stone-900 font-bold shadow-xs'
                          : 'text-stone-500 hover:text-stone-800'
                      }`}
                    >
                      {mode}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {/* Table Feed */}
            <div className="overflow-x-auto max-h-[460px] overflow-y-auto">
              <table className="w-full text-left border-collapse text-xs">
                <thead>
                  <tr className="border-b border-stone-200 text-stone-500 text-[11px] uppercase tracking-wider sticky top-0 bg-white">
                    <th className="py-2 px-3 font-semibold">Status</th>
                    <th className="py-2 px-3 font-semibold">Probe ID</th>
                    <th className="py-2 px-3 font-semibold">Time</th>
                    <th className="py-2 px-3 font-semibold">Latency</th>
                    <th className="py-2 px-3 font-semibold">HTTP Code</th>
                    <th className="py-2 px-3 font-semibold">Score / &tau;</th>
                    <th className="py-2 px-3 font-semibold">Classification</th>
                    <th className="py-2 px-3 font-semibold">Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-stone-100">
                  {filteredHistory.length === 0 ? (
                    <tr>
                      <td colSpan={8} className="py-12 text-center text-stone-400">
                        {isMonitoring
                          ? 'Waiting for first probe telemetry...'
                          : 'Stream is idle. Click "Start Reading & Detecting" above to populate live telemetry.'}
                      </td>
                    </tr>
                  ) : (
                    filteredHistory.map((item) => (
                      <tr
                        key={item.flow_id}
                        className={`transition-colors ${
                          item.severity === 'CRITICAL'
                            ? 'bg-rose-50/50 hover:bg-rose-50'
                            : item.is_anomaly
                            ? 'bg-amber-50/50 hover:bg-amber-50'
                            : 'hover:bg-stone-50/80'
                        }`}
                      >
                        <td className="py-2 px-3">
                          <span
                            className={`inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider ${
                              item.severity === 'CRITICAL'
                                ? 'bg-rose-100 text-rose-800 border border-rose-200'
                                : item.is_anomaly
                                ? 'bg-amber-100 text-amber-800 border border-amber-200'
                                : 'bg-emerald-100 text-emerald-800'
                            }`}
                          >
                            {item.is_anomaly ? 'ANOMALY' : 'NOMINAL'}
                          </span>
                        </td>
                        <td className="py-2 px-3 font-mono font-medium text-stone-800">
                          {item.flow_id_str}
                        </td>
                        <td className="py-2 px-3 font-mono text-stone-500">{item.timestamp}</td>
                        <td className="py-2 px-3 font-mono font-bold text-stone-900">
                          {item.latency_ms} ms
                        </td>
                        <td className="py-2 px-3 font-mono">
                          <span
                            className={`px-1.5 py-0.5 rounded text-[10px] font-semibold ${
                              item.status_code === 200
                                ? 'bg-emerald-50 text-emerald-700'
                                : item.status_code >= 500
                                ? 'bg-rose-100 text-rose-800'
                                : 'bg-amber-50 text-amber-800'
                            }`}
                          >
                            {item.status_code}
                          </span>
                        </td>
                        <td className="py-2 px-3 font-mono">
                          <span className={item.is_anomaly ? 'text-rose-600 font-bold' : 'text-stone-800'}>
                            {item.score.toFixed(2)}
                          </span>{' '}
                          <span className="text-stone-400 text-[10px]">/ {item.threshold.toFixed(2)}</span>
                        </td>
                        <td className="py-2 px-3 text-stone-700 max-w-[140px] truncate" title={item.threat_type}>
                          {item.threat_type}
                        </td>
                        <td className="py-2 px-3 text-stone-500 text-[11px] max-w-[150px] truncate" title={item.action}>
                          {item.action}
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>

            {/* Table Footer */}
            <div className="flex items-center justify-between text-xs text-stone-500 pt-3 border-t border-stone-100">
              <span>Showing {filteredHistory.length} live telemetry records in buffer</span>
              <button
                onClick={handleExportCSV}
                className="flex items-center space-x-1.5 text-stone-700 hover:text-stone-900 font-medium"
              >
                <Download className="w-3.5 h-3.5" />
                <span>Export Telemetry CSV</span>
              </button>
            </div>
          </section>
        </div>
      </main>
    </div>
  );
};
