"use client";

import { useApp } from "@/context/AppContext";

export default function PredictionBadge() {
  const { state } = useApp();
  if (!state.explainResult) return null;

  const { label, anomaly_score, model_type } = state.explainResult.prediction;
  const isAnomaly = label === "anomaly";

  return (
    <div
      className={`flex items-center justify-between rounded-lg px-5 py-4 ${
        isAnomaly ? "bg-red-50 ring-1 ring-red-200" : "bg-green-50 ring-1 ring-green-200"
      }`}
    >
      <div>
        <p className="text-xs font-medium uppercase tracking-wide text-gray-500">
          Prediction
        </p>
        <p
          className={`mt-0.5 text-2xl font-bold ${
            isAnomaly ? "text-red-600" : "text-green-600"
          }`}
        >
          {label.toUpperCase()}
        </p>
        <p className="mt-1 text-xs text-gray-500">
          Model: <span className="font-mono">{model_type}</span>
        </p>
      </div>
      <div className="text-right">
        <p className="text-xs font-medium uppercase tracking-wide text-gray-500">
          Anomaly Score
        </p>
        <p
          className={`mt-0.5 text-3xl font-bold tabular-nums ${
            isAnomaly ? "text-red-600" : "text-green-600"
          }`}
        >
          {anomaly_score.toFixed(4)}
        </p>
        <p className="mt-1 text-xs text-gray-400">
          {isAnomaly ? "Above threshold" : "Below threshold"}
        </p>
      </div>
    </div>
  );
}
