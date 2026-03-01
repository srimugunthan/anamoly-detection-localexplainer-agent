import type { ExplainResponse, Schema, UploadResponse } from "@/types";

const API_BASE = "/api";

export async function uploadModel(
  modelFile: File,
  schema: Schema,
  referenceCSV?: File | null
): Promise<UploadResponse> {
  const form = new FormData();
  form.append("model_file", modelFile);
  form.append("schema", JSON.stringify(schema));
  if (referenceCSV) {
    form.append("reference_csv", referenceCSV);
  }

  const res = await fetch(`${API_BASE}/upload-model`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "Upload failed");
  }
  return res.json();
}

export async function explainRecord(
  sessionId: string,
  inputRecord: Record<string, number | string>
): Promise<ExplainResponse> {
  const res = await fetch(`${API_BASE}/explain`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, input_record: inputRecord }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "Explain failed");
  }
  return res.json();
}

export function getPlotUrl(plotUrl: string): string {
  // plotUrl is already a relative path like /api/plot/{id}
  return plotUrl;
}
