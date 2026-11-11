// Type definitions for NETGUARD AI

export interface DatasetSummary {
  loaded: boolean;
  filename: string | null;
  total_flows: number;
  total_features: number;
  numerical_features: number;
  categorical_features: number;
  memory_usage_mb: number;
  missing_values_total: number;
  infinite_values_total: number;
  duplicate_rows: number;
  labels: Record<string, number>;
  anomaly_ratio: number;
  feature_names: string[];
}

export interface FeatureMeta {
  name: string;
  dtype: string;
  mean?: number;
  std?: number;
  min?: number;
  max?: number;
  missing: number;
  variance?: number;
}

export interface SignalGroup {
  id: string;
  name: string;
  category: string;
  description: string;
  features: string[];
}

export interface DistributionData {
  feature: string;
  bins: { bin: string; count: number }[];
  fences: {
    q1: number;
    median: number;
    q3: number;
    iqr: number;
    lower_fence: number;
    upper_fence: number;
    outlier_count: number;
    outlier_ratio: number;
  };
}

export interface CorrelationMatrix {
  features: string[];
  matrix: number[][];
}

export interface FlowAuditItem {
  flow_id: number;
  flow_id_str?: string;
  step?: number;
  timestamp: string;
  reconstruction_error: number;
  latent_distance: number;
  latent_deviation?: number;
  anomaly_score: number;
  threshold: number;
  is_anomaly: boolean;
  classification?: string;
  severity: 'NORMAL' | 'WARNING' | 'CRITICAL';
  ground_truth?: number;
  ground_truth_label?: string;
  destination_port?: number;
  attack_type?: string;
  recommended_action?: string;
}

export interface DetectionResult {
  method: string;
  total_evaluated: number;
  flagged_anomalies: number;
  normal_flows?: number;
  anomaly_rate: number;
  detection_ratio_pct?: number;
  mean_score: number;
  max_score: number;
  final_threshold: number;
  metrics: {
    f1_score: number;
    precision: number;
    recall: number;
    accuracy: number;
    mean_threshold?: number;
    min_threshold?: number;
    max_threshold?: number;
  };
  sample_points: Array<{
    step: number;
    timestamp: string;
    anomaly_score: number;
    threshold: number;
    is_anomaly: boolean;
    ground_truth?: number;
  }>;
  recent_flows: FlowAuditItem[];
  all_flows?: FlowAuditItem[];
  top_detections?: FlowAuditItem[];
}

export interface ThresholdComparisonResponse {
  trajectory: Array<{
    step: number;
    timestamp: string;
    anomaly_score: number;
    fixed_threshold: number;
    original_adaptive: number;
    stabilized_adaptive: number;
    is_burst: boolean;
    ground_truth: number;
    flagged_fixed: boolean;
    flagged_original: boolean;
    flagged_stabilized: boolean;
  }>;
  metrics: {
    fixed: ThresholdMethodMetric;
    original_adaptive: ThresholdMethodMetric;
    stabilized_adaptive: ThresholdMethodMetric;
  };
  simulation_params: {
    window_size: number;
    k_factor: number;
    alpha_smooth: number;
    inject_burst: boolean;
    burst_window: [number, number] | null;
  };
}

export interface ThresholdMethodMetric {
  name: string;
  offline_f1: number;
  threshold_variance: number;
  runaway_risk: string;
  description: string;
}

export interface ModelBenchmark {
  id: string;
  name: string;
  category: string;
  f1_score: number;
  precision: number;
  recall: number;
  accuracy: number;
  latency_ms: number;
  supervision: string;
  streaming_ready: boolean;
  notes: string;
  confusion_matrix: {
    tp: number;
    fp: number;
    fn: number;
    tn: number;
  };
}

export interface AutoencoderLayer {
  layer: number;
  units: number | string;
  activation: string;
  dropout: number;
}
