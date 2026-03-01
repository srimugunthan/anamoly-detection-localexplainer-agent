"use client";

import { useRef, useState, type DragEvent, type ChangeEvent } from "react";
import { uploadModel } from "@/api/client";
import { useApp } from "@/context/AppContext";
import type { Schema } from "@/types";

const EXAMPLE_SCHEMA = JSON.stringify(
  {
    features: {
      duration: { type: "float", min: 0, max: 5000 },
      amount: { type: "float", min: 0, max: 25000 },
      n_transactions: { type: "int", min: 0, max: 200 },
      category: { type: "categorical", values: ["retail", "online", "atm"] },
    },
  },
  null,
  2
);

export default function ModelUpload() {
  const { state, dispatch } = useApp();
  const [dragging, setDragging] = useState(false);
  const [modelFile, setModelFile] = useState<File | null>(null);
  const [schemaText, setSchemaText] = useState("");
  const [schemaError, setSchemaError] = useState<string | null>(null);
  const [referenceCSV, setReferenceCSV] = useState<File | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const csvInputRef = useRef<HTMLInputElement>(null);

  function parseSchema(text: string): Schema | null {
    try {
      const parsed = JSON.parse(text);
      if (!parsed.features || typeof parsed.features !== "object") {
        setSchemaError('Schema must have a "features" object.');
        return null;
      }
      setSchemaError(null);
      return parsed as Schema;
    } catch {
      setSchemaError("Invalid JSON.");
      return null;
    }
  }

  function onDrop(e: DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) setModelFile(file);
  }

  function onFileChange(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (file) setModelFile(file);
  }

  async function handleUpload() {
    const schema = parseSchema(schemaText);
    if (!modelFile || !schema) return;

    dispatch({ type: "UPLOAD_START" });
    try {
      const res = await uploadModel(modelFile, schema, referenceCSV);
      dispatch({
        type: "UPLOAD_SUCCESS",
        payload: {
          sessionId: res.session_id,
          schema,
          modelType: res.model_type,
          featureCount: res.feature_count,
        },
      });
    } catch (err) {
      dispatch({ type: "UPLOAD_ERROR", payload: String(err instanceof Error ? err.message : err) });
    }
  }

  const canUpload =
    modelFile !== null && schemaText.trim() !== "" && schemaError === null;

  return (
    <div className="space-y-4">
      <h2 className="text-sm font-semibold uppercase tracking-wide text-gray-500">
        1. Upload Model
      </h2>

      {/* Drop zone */}
      <div
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => fileInputRef.current?.click()}
        className={`flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed p-6 transition-colors ${
          dragging
            ? "border-indigo-400 bg-indigo-50"
            : modelFile
            ? "border-green-400 bg-green-50"
            : "border-gray-300 bg-gray-50 hover:border-indigo-300 hover:bg-indigo-50"
        }`}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".pkl,.joblib"
          className="hidden"
          onChange={onFileChange}
        />
        {modelFile ? (
          <p className="text-sm font-medium text-green-700">{modelFile.name}</p>
        ) : (
          <>
            <p className="text-sm text-gray-600">
              Drop <span className="font-medium">.pkl</span> or{" "}
              <span className="font-medium">.joblib</span> here
            </p>
            <p className="mt-1 text-xs text-gray-400">or click to browse</p>
          </>
        )}
      </div>

      {/* Schema editor */}
      <div>
        <div className="mb-1 flex items-center justify-between">
          <label className="text-xs font-medium text-gray-600">
            Feature Schema (JSON)
          </label>
          <button
            type="button"
            className="text-xs text-indigo-500 hover:underline"
            onClick={() => setSchemaText(EXAMPLE_SCHEMA)}
          >
            Load example
          </button>
        </div>
        <textarea
          value={schemaText}
          onChange={(e) => {
            setSchemaText(e.target.value);
            if (e.target.value.trim()) parseSchema(e.target.value);
            else setSchemaError(null);
          }}
          rows={8}
          placeholder={EXAMPLE_SCHEMA}
          className={`w-full rounded-md border px-3 py-2 font-mono text-xs focus:outline-none focus:ring-2 focus:ring-indigo-300 ${
            schemaError ? "border-red-400" : "border-gray-300"
          }`}
        />
        {schemaError && (
          <p className="mt-1 text-xs text-red-500">{schemaError}</p>
        )}
      </div>

      {/* Optional reference CSV */}
      <div>
        <label className="mb-1 block text-xs font-medium text-gray-600">
          Reference Data CSV{" "}
          <span className="font-normal text-gray-400">(optional, improves SHAP/LIME)</span>
        </label>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => csvInputRef.current?.click()}
            className="rounded border border-gray-300 px-3 py-1.5 text-xs text-gray-700 hover:bg-gray-50"
          >
            {referenceCSV ? referenceCSV.name : "Choose CSV…"}
          </button>
          {referenceCSV && (
            <button
              type="button"
              onClick={() => setReferenceCSV(null)}
              className="text-xs text-red-400 hover:underline"
            >
              Remove
            </button>
          )}
        </div>
        <input
          ref={csvInputRef}
          type="file"
          accept=".csv"
          className="hidden"
          onChange={(e) => setReferenceCSV(e.target.files?.[0] ?? null)}
        />
      </div>

      {/* Upload button */}
      <button
        type="button"
        disabled={!canUpload || state.uploadStatus === "uploading"}
        onClick={handleUpload}
        className="w-full rounded-md bg-indigo-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-indigo-700 disabled:cursor-not-allowed disabled:opacity-50"
      >
        {state.uploadStatus === "uploading" ? "Uploading…" : "Upload & Register Model"}
      </button>

      {/* Error */}
      {state.uploadStatus === "error" && state.appErrors.length > 0 && (
        <p className="rounded bg-red-50 px-3 py-2 text-xs text-red-600">
          {state.appErrors[0]}
        </p>
      )}

      {/* Success badge */}
      {state.uploadStatus === "success" && (
        <div className="rounded bg-green-50 px-3 py-2 text-xs text-green-700">
          <span className="font-medium">Ready.</span> Model:{" "}
          <span className="font-mono">{state.modelType}</span> &middot;{" "}
          {state.featureCount} features
          <button
            type="button"
            className="ml-3 text-indigo-500 hover:underline"
            onClick={() => dispatch({ type: "RESET" })}
          >
            Reset
          </button>
        </div>
      )}
    </div>
  );
}
