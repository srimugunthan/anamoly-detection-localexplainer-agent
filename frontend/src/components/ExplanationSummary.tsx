"use client";

import { useApp } from "@/context/AppContext";

const IMPACT_COLOR: Record<string, string> = {
  high: "bg-red-100 text-red-700",
  medium: "bg-yellow-100 text-yellow-700",
  low: "bg-green-100 text-green-700",
};

export default function ExplanationSummary() {
  const { state } = useApp();
  if (!state.explainResult) return null;

  const { text, feature_contributions } = state.explainResult.summary;
  const { errors } = state.explainResult;

  if (!text && !feature_contributions && errors.length === 0) return null;

  return (
    <div className="space-y-4">
      <h3 className="text-sm font-semibold uppercase tracking-wide text-gray-500">
        AI Explanation
      </h3>

      {/* Pipeline errors (non-fatal) */}
      {errors.length > 0 && (
        <div className="rounded-md bg-amber-50 px-4 py-3 text-xs text-amber-700">
          <p className="font-medium">Partial results — some tools encountered errors:</p>
          <ul className="mt-1 list-inside list-disc space-y-0.5">
            {errors.map((e, i) => (
              <li key={i}>{e}</li>
            ))}
          </ul>
        </div>
      )}

      {/* LLM summary text */}
      {text ? (
        <div className="rounded-md bg-indigo-50 px-4 py-3 text-sm leading-relaxed text-indigo-900">
          {text}
        </div>
      ) : (
        <p className="text-sm text-gray-400 italic">
          No LLM summary available (API key not configured or call failed).
        </p>
      )}

      {/* Feature contributions */}
      {feature_contributions && feature_contributions.length > 0 && (
        <div>
          <p className="mb-2 text-xs font-medium text-gray-500">Top contributors:</p>
          <div className="space-y-2">
            {feature_contributions.map((fc, i) => (
              <div
                key={i}
                className="flex items-start gap-3 rounded-md border border-gray-100 bg-white px-3 py-2"
              >
                <span
                  className={`mt-0.5 rounded px-1.5 py-0.5 text-xs font-medium ${
                    IMPACT_COLOR[fc.impact] ?? "bg-gray-100 text-gray-600"
                  }`}
                >
                  {fc.impact}
                </span>
                <div>
                  <p className="text-xs font-semibold text-gray-800">
                    {fc.feature}{" "}
                    <span className="font-normal text-gray-500">({fc.direction})</span>
                  </p>
                  <p className="text-xs text-gray-600">{fc.reason}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
