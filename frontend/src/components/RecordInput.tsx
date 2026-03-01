"use client";

import { useState, type FormEvent } from "react";
import { explainRecord } from "@/api/client";
import { useApp } from "@/context/AppContext";
import type { CategoricalFeatureDef, FloatFeatureDef } from "@/types";

export default function RecordInput() {
  const { state, dispatch } = useApp();
  const [values, setValues] = useState<Record<string, string>>({});
  const [jsonMode, setJsonMode] = useState(false);
  const [jsonText, setJsonText] = useState("");
  const [jsonError, setJsonError] = useState<string | null>(null);

  if (!state.schema || state.uploadStatus !== "success") return null;

  const features = Object.entries(state.schema.features);

  function setValue(name: string, val: string) {
    setValues((prev) => ({ ...prev, [name]: val }));
  }

  function buildRecord(): Record<string, number | string> | null {
    if (jsonMode) {
      try {
        const parsed = JSON.parse(jsonText);
        setJsonError(null);
        return parsed;
      } catch {
        setJsonError("Invalid JSON");
        return null;
      }
    }
    const record: Record<string, number | string> = {};
    for (const [name, def] of features) {
      const raw = values[name] ?? "";
      if (def.type === "categorical") {
        record[name] = raw;
      } else {
        const n = parseFloat(raw);
        if (isNaN(n)) return null;
        record[name] = n;
      }
    }
    return record;
  }

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    const record = buildRecord();
    if (!record || !state.sessionId) return;

    dispatch({ type: "EXPLAIN_START" });
    try {
      const res = await explainRecord(state.sessionId, record);
      dispatch({ type: "EXPLAIN_SUCCESS", payload: res });
    } catch (err) {
      dispatch({ type: "EXPLAIN_ERROR", payload: String(err instanceof Error ? err.message : err) });
    }
  }

  const allFilled = jsonMode
    ? jsonText.trim() !== ""
    : features.every(([name]) => (values[name] ?? "").trim() !== "");

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-gray-500">
          2. Input Record
        </h2>
        <button
          type="button"
          className="text-xs text-indigo-500 hover:underline"
          onClick={() => {
            setJsonMode(!jsonMode);
            setJsonError(null);
          }}
        >
          {jsonMode ? "Form view" : "JSON paste"}
        </button>
      </div>

      {jsonMode ? (
        <div>
          <textarea
            value={jsonText}
            onChange={(e) => setJsonText(e.target.value)}
            rows={6}
            placeholder={`{\n  "duration": 120,\n  "amount": 350.0\n}`}
            className={`w-full rounded-md border px-3 py-2 font-mono text-xs focus:outline-none focus:ring-2 focus:ring-indigo-300 ${
              jsonError ? "border-red-400" : "border-gray-300"
            }`}
          />
          {jsonError && (
            <p className="mt-1 text-xs text-red-500">{jsonError}</p>
          )}
        </div>
      ) : (
        <div className="space-y-3">
          {features.map(([name, def]) => (
            <div key={name}>
              <label className="mb-1 block text-xs font-medium text-gray-700">
                {name}
                <span className="ml-1 text-gray-400">({def.type})</span>
              </label>
              {def.type === "categorical" ? (
                <select
                  value={values[name] ?? ""}
                  onChange={(e) => setValue(name, e.target.value)}
                  className="w-full rounded-md border border-gray-300 px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-300"
                >
                  <option value="">Select…</option>
                  {(def as CategoricalFeatureDef).values.map((v) => (
                    <option key={v} value={v}>
                      {v}
                    </option>
                  ))}
                </select>
              ) : (
                <input
                  type="number"
                  step="any"
                  min={(def as FloatFeatureDef).min}
                  max={(def as FloatFeatureDef).max}
                  placeholder={`${(def as FloatFeatureDef).min} – ${(def as FloatFeatureDef).max}`}
                  value={values[name] ?? ""}
                  onChange={(e) => setValue(name, e.target.value)}
                  className="w-full rounded-md border border-gray-300 px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-300"
                />
              )}
            </div>
          ))}
        </div>
      )}

      <button
        type="submit"
        disabled={!allFilled || state.isExplaining}
        className="w-full rounded-md bg-indigo-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-indigo-700 disabled:cursor-not-allowed disabled:opacity-50"
      >
        {state.isExplaining ? "Explaining…" : "Explain"}
      </button>

      {state.isExplaining === false &&
        state.appErrors.length > 0 &&
        state.explainResult === null && (
          <p className="rounded bg-red-50 px-3 py-2 text-xs text-red-600">
            {state.appErrors[0]}
          </p>
        )}
    </form>
  );
}
