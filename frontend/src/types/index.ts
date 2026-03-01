// ─── Schema ───────────────────────────────────────────────────────────────────

export interface FloatFeatureDef {
  type: "float" | "int";
  min: number;
  max: number;
}

export interface CategoricalFeatureDef {
  type: "categorical";
  values: string[];
}

export type FeatureDef = FloatFeatureDef | CategoricalFeatureDef;

export interface Schema {
  features: Record<string, FeatureDef>;
}

// ─── Upload ───────────────────────────────────────────────────────────────────

export interface UploadResponse {
  session_id: string;
  model_type: string;
  feature_count: number;
  status: string;
}

// ─── Explain ─────────────────────────────────────────────────────────────────

export interface PlotURLs {
  shap_plot_url: string | null;
  shap_force_plot_url: string | null;
  lime_plot_url: string | null;
  pdp_plot_urls: string[];
}

export interface PredictionResult {
  label: string;
  anomaly_score: number;
  model_type: string;
}

export interface ExplanationsResult {
  shap_values: Record<string, number> | null;
  lime_weights: Record<string, number> | null;
  top_features: string[] | null;
  plots: PlotURLs;
}

export interface FeatureContribution {
  feature: string;
  impact: "high" | "medium" | "low";
  direction: string;
  reason: string;
}

export interface SummaryResult {
  text: string | null;
  feature_contributions: FeatureContribution[] | null;
}

export interface ExplainResponse {
  prediction: PredictionResult;
  explanations: ExplanationsResult;
  summary: SummaryResult;
  errors: string[];
}

// ─── App state ────────────────────────────────────────────────────────────────

export type UploadStatus = "idle" | "uploading" | "success" | "error";

export interface AppState {
  sessionId: string | null;
  schema: Schema | null;
  modelType: string | null;
  featureCount: number | null;
  uploadStatus: UploadStatus;
  explainResult: ExplainResponse | null;
  isExplaining: boolean;
  appErrors: string[];
}
