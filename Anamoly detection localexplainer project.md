# Anomaly Detection Local Explainer Agent

## Overview

An AI-powered agent that provides **local explanations** for anomaly detection model predictions. The user uploads a trained model object and defines the input data schema via a UI. For any given input record (JSON), the agent runs the model's `.predict()` method, invokes explainability tools (SHAP, LIME, PDP), uses an LLM to interpret the generated plots, and surfaces a consolidated natural-language explanation back in the UI.

---

## Problem Statement

Anomaly detection models (Isolation Forest, Autoencoders, One-Class SVM, etc.) are often black boxes. When a record is flagged as anomalous (or normal), stakeholders need to understand **why** — which features contributed, in what direction, and by how much. Existing explainability libraries produce plots, but interpreting them requires ML expertise. This agent **closes the last mile** by automatically interpreting the visual outputs and generating human-readable explanations.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        FRONTEND (UI)                        │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐ │
│  │ Upload Model  │  │ Define Schema│  │ Input JSON Record │ │
│  │ (.pkl/.joblib)│  │ (col types)  │  │ + Submit Button   │ │
│  └──────┬───────┘  └──────┬───────┘  └────────┬──────────┘ │
│         │                 │                    │            │
│         ▼                 ▼                    ▼            │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Explanation Summary Panel               │    │
│  │  • Prediction result (anomaly / normal + score)     │    │
│  │  • SHAP interpretation                              │    │
│  │  • LIME interpretation                              │    │
│  │  • PDP interpretation                               │    │
│  │  • Consolidated natural-language summary             │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────┬───────────────────────────────────┘
                          │  REST / WebSocket
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    BACKEND (FastAPI)                         │
│                                                             │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                  LangGraph Agent                       │ │
│  │                                                        │ │
│  │  ┌──────────┐    ┌────────────┐    ┌───────────────┐  │ │
│  │  │ Predict  │───▶│ Explain    │───▶│ Interpret &   │  │ │
│  │  │ Node     │    │ Node       │    │ Summarize Node│  │ │
│  │  └──────────┘    └────────────┘    └───────────────┘  │ │
│  │       │               │                    │          │ │
│  │       ▼               ▼                    ▼          │ │
│  │  model.predict() ┌─────────┐         LLM (Gemini/    │ │
│  │                   │  Tools  │         Claude) call     │ │
│  │                   │─────────│         with plot images │ │
│  │                   │ • SHAP  │                          │ │
│  │                   │ • LIME  │                          │ │
│  │                   │ • PDP   │                          │ │
│  │                   └─────────┘                          │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## Core Workflow

### Step 1 — Model & Schema Upload (UI)

| Input          | Description                                                                 |
|----------------|-----------------------------------------------------------------------------|
| **Model file** | A serialized scikit-learn-compatible model (`.pkl`, `.joblib`, or ONNX).    |
| **Schema**     | JSON schema defining feature names, data types, and allowed value ranges.   |

**Schema example:**

```json
{
  "features": [
    {"name": "transaction_amount", "type": "float", "description": "USD amount"},
    {"name": "hour_of_day", "type": "int", "description": "0-23"},
    {"name": "merchant_category", "type": "categorical", "values": ["retail", "food", "travel", "online"]},
    {"name": "distance_from_home", "type": "float", "description": "miles"}
  ]
}
```

### Step 2 — Submit Input Record (UI)

User provides a single JSON record to explain:

```json
{
  "transaction_amount": 4500.00,
  "hour_of_day": 3,
  "merchant_category": "online",
  "distance_from_home": 1200.5
}
```

### Step 3 — Prediction (Agent → Predict Node)

- Load the model object into memory.
- Validate the input record against the schema.
- Run `model.predict(input_df)` and (if available) `model.decision_function(input_df)` or `model.score_samples(input_df)`.
- Return: **prediction label** (anomaly/normal) + **anomaly score**.

### Step 4 — Explainability Tool Invocation (Agent → Explain Node)

The agent invokes one or more of the following **tools** (LangGraph tool nodes):

#### Tool 1: SHAP (Local Feature Attribution)

- Use `shap.KernelExplainer` (model-agnostic) or `shap.TreeExplainer` (tree-based models).
- Generate a **SHAP waterfall plot** for the single input record.
- Generate a **SHAP force plot** as supplementary view.
- Save plots as PNG/SVG to a temp directory.

#### Tool 2: LIME (Local Surrogate Model)

- Use `lime.lime_tabular.LimeTabularExplainer`.
- Fit a local surrogate on a neighborhood of the input record.
- Generate the **LIME explanation bar chart** showing feature contributions.
- Save plot as PNG/SVG.

#### Tool 3: PDP / ICE (Global Context for Local Point)

- Use `sklearn.inspection.PartialDependenceDisplay` or custom ICE implementation.
- Generate **PDP plots** for the top-K influential features (identified from SHAP/LIME).
- Overlay the current input record's feature value on each PDP curve.
- Save plots as PNG/SVG.

### Step 5 — LLM Interpretation (Agent → Interpret & Summarize Node)

- Collect all generated plot images + numerical outputs (SHAP values, LIME weights).
- Send to LLM (Gemini / Claude) with a structured prompt:

```
You are an ML explainability expert. Given:
- Model type: {model_type}
- Prediction: {prediction_label} (score: {anomaly_score})
- SHAP values: {shap_values_dict}
- LIME weights: {lime_weights_dict}
- Attached plots: [SHAP waterfall, LIME bar chart, PDP plots]

Provide:
1. A plain-English summary of WHY this record was flagged as {prediction_label}.
2. The top 3 contributing features and their impact direction.
3. Any feature interactions or non-linear effects visible in the PDP plots.
4. A confidence assessment of the explanation consistency across SHAP and LIME.
5. Suggested next steps for the analyst.
```

### Step 6 — Display Summary (UI)

The UI renders:
- **Prediction badge** — Anomaly (red) / Normal (green) with score.
- **Plots section** — Tabbed view of SHAP, LIME, PDP charts (interactive where possible).
- **AI Explanation panel** — The LLM-generated natural-language summary.
- **Feature contribution table** — Sortable table of features ranked by importance.

---

## LangGraph Agent Design

### State Schema

```python
from typing import TypedDict, Optional
from langgraph.graph import StateGraph

class ExplainerState(TypedDict):
    # Inputs
    model_bytes: bytes
    schema: dict
    input_record: dict

    # Prediction
    prediction_label: str          # "anomaly" or "normal"
    anomaly_score: float
    model_type: str                # e.g., "IsolationForest"

    # Explainability outputs
    shap_values: Optional[dict]
    shap_plot_path: Optional[str]
    lime_weights: Optional[dict]
    lime_plot_path: Optional[str]
    pdp_plot_paths: Optional[list[str]]
    top_features: Optional[list[str]]

    # LLM interpretation
    explanation_summary: Optional[str]
    feature_contributions: Optional[list[dict]]
    errors: list[str]
```

### Graph Topology

```
                 ┌──────────┐
                 │  START    │
                 └────┬─────┘
                      ▼
              ┌───────────────┐
              │ validate_and  │
              │ _load_model   │
              └───────┬───────┘
                      ▼
              ┌───────────────┐
              │   predict     │
              └───────┬───────┘
                      ▼
         ┌────────────┼────────────┐
         ▼            ▼            ▼
   ┌──────────┐ ┌──────────┐ ┌──────────┐
   │  shap    │ │  lime    │ │  pdp     │
   │  tool    │ │  tool    │ │  tool    │
   └────┬─────┘ └────┬─────┘ └────┬─────┘
         └────────────┼────────────┘
                      ▼
              ┌───────────────┐
              │  interpret    │
              │  (LLM call)   │
              └───────┬───────┘
                      ▼
              ┌───────────────┐
              │   END         │
              └───────────────┘
```

- **SHAP, LIME, PDP run in parallel** (LangGraph fan-out) for latency optimization.
- Each tool node writes results back to shared state.
- The `interpret` node aggregates all outputs and calls the LLM.

---

## Tech Stack

| Layer            | Technology                                                |
|------------------|-----------------------------------------------------------|
| **Frontend**     | React / Next.js with Tailwind CSS                         |
| **Backend API**  | FastAPI (async)                                           |
| **Agent**        | LangGraph (StateGraph with parallel tool nodes)           |
| **LLM**         | Gemini API (via custom HTTP wrapper) or Claude API        |
| **SHAP**        | `shap` Python library                                     |
| **LIME**        | `lime` Python library                                     |
| **PDP**         | `scikit-learn` inspection module + `matplotlib`           |
| **Model I/O**   | `joblib` / `pickle` / `onnxruntime`                       |
| **Plot Render**  | `matplotlib` + `plotly` (for interactive UI charts)       |
| **Storage**      | Temp file system (plots), in-memory (model + data)        |

---

## API Endpoints

### `POST /api/upload-model`

Upload the model file and schema.

**Request:** `multipart/form-data`
- `model_file`: binary (pkl/joblib)
- `schema`: JSON string

**Response:**
```json
{
  "session_id": "uuid",
  "model_type": "IsolationForest",
  "feature_count": 4,
  "status": "ready"
}
```

### `POST /api/explain`

Submit a record for explanation.

**Request:**
```json
{
  "session_id": "uuid",
  "input_record": {
    "transaction_amount": 4500.00,
    "hour_of_day": 3,
    "merchant_category": "online",
    "distance_from_home": 1200.5
  },
  "tools": ["shap", "lime", "pdp"]
}
```

**Response:**
```json
{
  "prediction": {
    "label": "anomaly",
    "score": -0.72
  },
  "explanations": {
    "shap": {
      "values": {"transaction_amount": 0.45, "hour_of_day": 0.28, ...},
      "plot_url": "/plots/shap_waterfall_abc123.png"
    },
    "lime": {
      "weights": {"transaction_amount > 3000": 0.38, ...},
      "plot_url": "/plots/lime_bar_abc123.png"
    },
    "pdp": {
      "features": ["transaction_amount", "hour_of_day"],
      "plot_urls": ["/plots/pdp_txn_abc123.png", "/plots/pdp_hour_abc123.png"]
    }
  },
  "summary": {
    "text": "This transaction was flagged as anomalous primarily due to ...",
    "top_contributors": [
      {"feature": "transaction_amount", "impact": "high", "direction": "increases anomaly score"},
      {"feature": "hour_of_day", "impact": "medium", "direction": "unusual time window"},
      {"feature": "distance_from_home", "impact": "medium", "direction": "far from typical location"}
    ],
    "consistency_note": "SHAP and LIME agree on the top 2 features...",
    "next_steps": ["Verify with cardholder", "Check merchant history"]
  }
}
```

### `GET /api/plot/{plot_id}`

Serve generated plot images.

---

## Supported Model Types

| Model                          | `.predict()` | `.decision_function()` | `.score_samples()` | SHAP Explainer     |
|--------------------------------|:------------:|:----------------------:|:-------------------:|-------------------|
| Isolation Forest               | ✅           | ✅                     | ✅                  | TreeExplainer     |
| One-Class SVM                  | ✅           | ✅                     | ❌                  | KernelExplainer   |
| Local Outlier Factor           | ✅           | ❌                     | ✅                  | KernelExplainer   |
| Autoencoder (reconstruction)   | ✅*          | ❌                     | ❌                  | KernelExplainer   |
| Elliptic Envelope              | ✅           | ✅                     | ✅                  | KernelExplainer   |
| Custom `.predict()` models     | ✅           | varies                 | varies              | KernelExplainer   |

*Autoencoder: `.predict()` returns reconstruction error thresholded as anomaly/normal.

---

## Background Data Handling

SHAP and LIME require a **background/training dataset** to compute explanations. Options:

1. **User uploads a reference CSV** alongside the model (preferred for accuracy).
2. **Synthetic generation** — the agent generates synthetic samples from the schema (uniform/normal distribution per feature type).
3. **K-means summary** — if a large reference set is provided, summarize to K representative points for efficiency.

The UI should prompt the user:

> "For best explanations, upload a reference dataset (CSV) that represents normal behavior. Otherwise, synthetic samples will be generated from the schema."

---

## Error Handling

| Scenario                              | Handling                                                       |
|---------------------------------------|----------------------------------------------------------------|
| Model file corrupt / incompatible     | Return validation error with supported formats                 |
| Input record fails schema validation  | Return field-level errors with expected types/ranges           |
| SHAP/LIME timeout (>60s)             | Fall back to available tools; note partial explanation          |
| LLM API failure                       | Return raw SHAP/LIME values + plots without AI interpretation  |
| Model lacks `.predict()`              | Return error: "Model must implement a `.predict()` method"     |
| Categorical encoding mismatch         | Auto-detect encoder from model pipeline; error if not found    |

---

## Project Structure

```
anomaly-explainer-agent/
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ModelUpload.tsx
│   │   │   ├── SchemaEditor.tsx
│   │   │   ├── RecordInput.tsx
│   │   │   ├── PredictionBadge.tsx
│   │   │   ├── PlotViewer.tsx
│   │   │   └── ExplanationSummary.tsx
│   │   ├── pages/
│   │   │   └── index.tsx
│   │   └── api/
│   │       └── client.ts
│   └── package.json
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPI app
│   │   ├── routers/
│   │   │   ├── upload.py
│   │   │   └── explain.py
│   │   ├── agent/
│   │   │   ├── graph.py            # LangGraph agent definition
│   │   │   ├── state.py            # State schema
│   │   │   └── nodes/
│   │   │       ├── predict.py
│   │   │       ├── shap_tool.py
│   │   │       ├── lime_tool.py
│   │   │       ├── pdp_tool.py
│   │   │       └── interpret.py
│   │   ├── services/
│   │   │   ├── model_loader.py
│   │   │   ├── schema_validator.py
│   │   │   └── plot_generator.py
│   │   └── config.py
│   ├── tests/
│   │   ├── test_predict.py
│   │   ├── test_shap_tool.py
│   │   ├── test_lime_tool.py
│   │   └── test_agent_e2e.py
│   └── requirements.txt
├── notebooks/
│   └── demo_walkthrough.ipynb
├── project.md                      # This file
└── README.md
```

---

## Milestones

| Phase   | Deliverable                                    | Duration |
|---------|------------------------------------------------|----------|
| **M1**  | Model loader + schema validation + predict     | 1 week   |
| **M2**  | SHAP, LIME, PDP tool nodes (standalone)        | 1 week   |
| **M3**  | LangGraph agent wiring (parallel execution)    | 1 week   |
| **M4**  | LLM interpretation node + prompt engineering   | 1 week   |
| **M5**  | FastAPI endpoints + integration tests          | 1 week   |
| **M6**  | React UI (upload, input, plots, summary)       | 2 weeks  |
| **M7**  | End-to-end testing + edge cases + docs         | 1 week   |

---

## Key Design Decisions

1. **LangGraph over simple chaining** — Enables parallel execution of SHAP/LIME/PDP, structured state management, and easy addition of new explainability tools as nodes.

2. **Model-agnostic approach** — Using `KernelExplainer` as fallback ensures any model with `.predict()` works, even custom models.

3. **Plot-based LLM interpretation** — Sending both raw numerical values AND plot images to the LLM produces richer, more contextual explanations than numbers alone.

4. **Schema-driven validation** — Upfront schema definition prevents runtime errors and enables synthetic background data generation.

5. **Parallel tool execution** — SHAP, LIME, and PDP are independent computations; running them concurrently reduces total latency by ~60%.

---

## Future Enhancements

- **Batch explanation** — Explain multiple records at once with aggregated summaries.
- **Counterfactual explanations** — "What would need to change for this to be normal?"
- **Streaming UI** — Stream explanation results as each tool completes (SSE/WebSocket).
- **Model comparison** — Upload two models and compare explanations side-by-side.
- **Feedback loop** — Analyst marks explanations as helpful/unhelpful to improve prompts.
- **ONNX support** — Accept ONNX models for framework-agnostic deployment.
- **MCP integration** — Expose explainability tools via Model Context Protocol for use in other agent frameworks.
