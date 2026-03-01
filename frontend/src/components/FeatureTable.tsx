"use client";

import { useState } from "react";
import { useApp } from "@/context/AppContext";

type SortKey = "feature" | "shap" | "lime";
type SortDir = "asc" | "desc";

function ImpactBar({ value, max }: { value: number; max: number }) {
  const pct = max === 0 ? 0 : Math.min(100, (Math.abs(value) / max) * 100);
  const positive = value >= 0;
  return (
    <div className="flex items-center gap-2">
      <div className="h-2 w-24 overflow-hidden rounded-full bg-gray-100">
        <div
          className={`h-full rounded-full ${positive ? "bg-red-400" : "bg-blue-400"}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className={`text-xs tabular-nums ${positive ? "text-red-600" : "text-blue-600"}`}>
        {value > 0 ? "+" : ""}
        {value.toFixed(4)}
      </span>
    </div>
  );
}

export default function FeatureTable() {
  const { state } = useApp();
  const [sortKey, setSortKey] = useState<SortKey>("shap");
  const [sortDir, setSortDir] = useState<SortDir>("desc");

  if (!state.explainResult) return null;

  const { shap_values, lime_weights, top_features } = state.explainResult.explanations;
  if (!shap_values && !lime_weights) return null;

  // Build unified row set — union of all features
  const featureNames = Array.from(
    new Set([
      ...(top_features ?? []),
      ...Object.keys(shap_values ?? {}),
      ...Object.keys(lime_weights ?? {}),
    ])
  );

  const maxShap = Math.max(...Object.values(shap_values ?? {}).map(Math.abs), 0);
  const maxLime = Math.max(...Object.values(lime_weights ?? {}).map(Math.abs), 0);

  const rows = featureNames.map((f) => ({
    feature: f,
    shap: shap_values?.[f] ?? null,
    lime: lime_weights?.[f] ?? null,
  }));

  const sorted = [...rows].sort((a, b) => {
    let aVal: number, bVal: number;
    if (sortKey === "feature") {
      return sortDir === "asc"
        ? a.feature.localeCompare(b.feature)
        : b.feature.localeCompare(a.feature);
    }
    aVal = Math.abs(a[sortKey] ?? 0);
    bVal = Math.abs(b[sortKey] ?? 0);
    return sortDir === "asc" ? aVal - bVal : bVal - aVal;
  });

  function toggleSort(key: SortKey) {
    if (sortKey === key) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      setSortDir("desc");
    }
  }

  function SortIcon({ col }: { col: SortKey }) {
    if (sortKey !== col) return <span className="text-gray-300"> ↕</span>;
    return <span className="text-indigo-500"> {sortDir === "asc" ? "↑" : "↓"}</span>;
  }

  return (
    <div>
      <h3 className="mb-3 text-sm font-semibold uppercase tracking-wide text-gray-500">
        Feature Contributions
      </h3>
      <div className="overflow-x-auto rounded-md border border-gray-200">
        <table className="w-full text-xs">
          <thead className="bg-gray-50 text-left text-gray-500">
            <tr>
              <th
                className="cursor-pointer px-3 py-2 font-medium hover:text-gray-800"
                onClick={() => toggleSort("feature")}
              >
                Feature
                <SortIcon col="feature" />
              </th>
              {shap_values && (
                <th
                  className="cursor-pointer px-3 py-2 font-medium hover:text-gray-800"
                  onClick={() => toggleSort("shap")}
                >
                  SHAP
                  <SortIcon col="shap" />
                </th>
              )}
              {lime_weights && (
                <th
                  className="cursor-pointer px-3 py-2 font-medium hover:text-gray-800"
                  onClick={() => toggleSort("lime")}
                >
                  LIME
                  <SortIcon col="lime" />
                </th>
              )}
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {sorted.map((row) => (
              <tr key={row.feature} className="hover:bg-gray-50">
                <td className="px-3 py-2 font-mono font-medium text-gray-800">
                  {row.feature}
                  {top_features?.includes(row.feature) && (
                    <span className="ml-1 rounded bg-yellow-100 px-1 text-yellow-700">
                      top
                    </span>
                  )}
                </td>
                {shap_values && (
                  <td className="px-3 py-2">
                    {row.shap !== null ? (
                      <ImpactBar value={row.shap} max={maxShap} />
                    ) : (
                      <span className="text-gray-300">—</span>
                    )}
                  </td>
                )}
                {lime_weights && (
                  <td className="px-3 py-2">
                    {row.lime !== null ? (
                      <ImpactBar value={row.lime} max={maxLime} />
                    ) : (
                      <span className="text-gray-300">—</span>
                    )}
                  </td>
                )}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
