"use client";

import { useApp } from "@/context/AppContext";

const STEPS = ["Predict", "SHAP", "LIME", "PDP", "Interpret"];

export default function LoadingState() {
  const { state } = useApp();
  if (!state.isExplaining) return null;

  return (
    <div className="space-y-5 py-4">
      {/* Step indicator */}
      <div>
        <p className="mb-3 text-xs font-semibold uppercase tracking-wide text-gray-500">
          Running explanation pipeline…
        </p>
        <div className="flex items-center gap-2">
          {STEPS.map((step, i) => (
            <div key={step} className="flex items-center gap-2">
              <div className="flex flex-col items-center">
                <div className="h-2.5 w-2.5 animate-pulse rounded-full bg-indigo-400" />
                <p className="mt-1 text-xs text-gray-500">{step}</p>
              </div>
              {i < STEPS.length - 1 && (
                <div className="mb-4 h-px w-6 bg-gray-200" />
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Skeleton cards */}
      <div className="animate-pulse space-y-3">
        <div className="h-20 rounded-lg bg-gray-100" />
        <div className="h-48 rounded-lg bg-gray-100" />
        <div className="h-32 rounded-lg bg-gray-100" />
      </div>
    </div>
  );
}
