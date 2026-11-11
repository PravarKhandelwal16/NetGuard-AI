import {
  DatasetSummary,
  FeatureMeta,
  SignalGroup,
  DistributionData,
  CorrelationMatrix,
  DetectionResult,
  ThresholdComparisonResponse,
  ModelBenchmark,
  AutoencoderLayer,
} from '../types';

const BASE_URL = '/api';

export const api = {
  // Dataset Endpoints
  async getDatasetSummary(): Promise<DatasetSummary> {
    const res = await fetch(`${BASE_URL}/dataset/summary`);
    if (!res.ok) throw new Error(`Failed to fetch dataset summary: ${res.statusText}`);
    return res.json();
  },

  async getDatasetFeatures(): Promise<{ total: number; features: FeatureMeta[] }> {
    const res = await fetch(`${BASE_URL}/dataset/features`);
    if (!res.ok) throw new Error(`Failed to fetch dataset features: ${res.statusText}`);
    return res.json();
  },

  async getDatasetPreview(limit: number = 50, offset: number = 0): Promise<{
    columns: string[];
    rows: Record<string, any>[];
    total_rows: number;
    limit: number;
    offset: number;
  }> {
    const res = await fetch(`${BASE_URL}/dataset/preview?limit=${limit}&offset=${offset}`);
    if (!res.ok) throw new Error(`Failed to preview dataset: ${res.statusText}`);
    return res.json();
  },

  async searchDataset(query: string, limit: number = 20): Promise<{
    total: number;
    query: string;
    results: Array<{
      flow_id: number;
      flow_id_str: string;
      label: string;
      is_anomaly: boolean;
      destination_port: number | null;
      protocol: string;
      timestamp: string;
      flow_bytes_s: number | null;
      flow_duration: number | null;
    }>;
  }> {
    const res = await fetch(`${BASE_URL}/dataset/search?query=${encodeURIComponent(query)}&limit=${limit}`);
    if (!res.ok) throw new Error(`Search failed: ${res.statusText}`);
    return res.json();
  },

  async uploadDataset(file: File): Promise<any> {
    const formData = new FormData();
    formData.append('file', file);
    const res = await fetch(`${BASE_URL}/dataset/upload`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) throw new Error(`File upload failed: ${res.statusText}`);
    return res.json();
  },

  async loadSampleDataset(name: string): Promise<any> {
    const res = await fetch(`${BASE_URL}/dataset/load-sample?name=${encodeURIComponent(name)}`, {
      method: 'POST',
    });
    if (!res.ok) throw new Error(`Sample load failed: ${res.statusText}`);
    return res.json();
  },

  // Traffic Analysis Endpoints
  async getSignalGroups(): Promise<{ groups: SignalGroup[] }> {
    const res = await fetch(`${BASE_URL}/traffic/groups`);
    if (!res.ok) throw new Error(`Failed to fetch signal groups: ${res.statusText}`);
    return res.json();
  },

  async getFeatureDistribution(feature: string, bins: number = 25): Promise<DistributionData> {
    const res = await fetch(`${BASE_URL}/traffic/distribution?feature=${encodeURIComponent(feature)}&bins=${bins}`);
    if (!res.ok) throw new Error(`Failed to fetch feature distribution: ${res.statusText}`);
    return res.json();
  },

  async getCorrelationMatrix(features?: string[]): Promise<CorrelationMatrix> {
    const query = features ? `?features=${encodeURIComponent(features.join(','))}` : '';
    const res = await fetch(`${BASE_URL}/traffic/correlation${query}`);
    if (!res.ok) throw new Error(`Failed to fetch correlation matrix: ${res.statusText}`);
    return res.json();
  },

  // Detection Endpoints
  async runDetection(params: {
    method?: string;
    alpha?: number;
    base_threshold?: number;
    window_size?: number;
    sample_limit?: number;
  }): Promise<DetectionResult> {
    const res = await fetch(`${BASE_URL}/detection/run`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    if (!res.ok) throw new Error(`Inference run failed: ${res.statusText}`);
    return res.json();
  },

  async getDetectionHistory(limit: number = 100): Promise<{ total: number; flows: any[] }> {
    const res = await fetch(`${BASE_URL}/detection/history?limit=${limit}`);
    if (!res.ok) throw new Error(`Failed to fetch history: ${res.statusText}`);
    return res.json();
  },

  // Threshold Endpoints
  async compareThresholds(params: {
    window_size?: number;
    k_factor?: number;
    base_tau?: number;
    alpha_smooth?: number;
    inject_burst?: boolean;
    stream_length?: number;
  }): Promise<ThresholdComparisonResponse> {
    const res = await fetch(`${BASE_URL}/threshold/compare`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    if (!res.ok) throw new Error(`Threshold comparison failed: ${res.statusText}`);
    return res.json();
  },

  // Models Leaderboard
  async getModelsLeaderboard(): Promise<{
    models: ModelBenchmark[];
    source?: string;
    architecture: {
      input_dim: number;
      encoder_layers: AutoencoderLayer[];
      decoder_layers: AutoencoderLayer[];
      loss: string;
      optimizer: string;
      training_data: string;
      dual_signal_formula: string;
    };
    summary: Record<string, string>;
  }> {
    const res = await fetch(`${BASE_URL}/models/leaderboard`);
    if (!res.ok) throw new Error(`Failed to fetch models leaderboard: ${res.statusText}`);
    return res.json();
  },

  // Pipeline Execution Endpoints
  async getPipelineStatus(): Promise<{
    status: string;
    artifacts: Record<string, any>;
    total_artifacts_found: number;
    total_expected: number;
  }> {
    const res = await fetch(`${BASE_URL}/pipeline/status`);
    if (!res.ok) throw new Error(`Failed to fetch pipeline status: ${res.statusText}`);
    return res.json();
  },

  async getPipelineResults(): Promise<{
    final_comparison: any[];
    attharv_results: any[];
    ishan_results: any[];
  }> {
    const res = await fetch(`${BASE_URL}/pipeline/results`);
    if (!res.ok) throw new Error(`Failed to fetch pipeline results: ${res.statusText}`);
    return res.json();
  },

  async runPipeline(datasetPath?: string): Promise<any> {
    const query = datasetPath ? `?dataset_path=${encodeURIComponent(datasetPath)}` : '';
    const res = await fetch(`${BASE_URL}/pipeline/run${query}`, {
      method: 'POST',
    });
    if (!res.ok) throw new Error(`Pipeline run failed: ${res.statusText}`);
    return res.json();
  },

  // Live Service Telemetry Endpoints
  async probeLiveService(params: {
    url: string;
    simulate_attack?: string | null;
    alpha?: number;
    base_threshold?: number;
    engine_mode?: string;
  }): Promise<{
    status: string;
    probe: {
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
    };
    metrics: {
      total_probes: number;
      anomalies_detected: number;
      critical_count: number;
      anomaly_rate_pct: number;
      avg_latency_ms: number;
      baseline_median_ms: number;
      current_threshold: number;
    };
    recent_window: any[];
    active_simulation: string | null;
  }> {
    const res = await fetch(`${BASE_URL}/live/probe`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    if (!res.ok) throw new Error(`Live probe failed: ${res.statusText}`);
    return res.json();
  },

  async setLiveSimulation(url: string, mode: string | null): Promise<any> {
    const query = mode ? `?url=${encodeURIComponent(url)}&mode=${encodeURIComponent(mode)}` : `?url=${encodeURIComponent(url)}`;
    const res = await fetch(`${BASE_URL}/live/set-simulation${query}`, {
      method: 'POST',
    });
    if (!res.ok) throw new Error(`Simulation toggle failed: ${res.statusText}`);
    return res.json();
  },

  async clearLiveHistory(url?: string): Promise<any> {
    const query = url ? `?url=${encodeURIComponent(url)}` : '';
    const res = await fetch(`${BASE_URL}/live/clear-history${query}`, {
      method: 'POST',
    });
    if (!res.ok) throw new Error(`Clear history failed: ${res.statusText}`);
    return res.json();
  },
};
