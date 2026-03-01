# Anomaly Explainer Agent

An AI-powered agent that explains anomaly detection model predictions using **SHAP**, **LIME**, and **PDP**, then synthesises results into plain-English summaries via an LLM (Claude or Gemini).

```
┌──────────────────────────────────────────────────────┐
│               Anomaly Explainer UI (Next.js)          │
│  ┌─────────────────┐   ┌──────────────────────────┐  │
│  │  ModelUpload    │   │  PredictionBadge         │  │
│  │  SchemaEditor   │   │  ExplanationSummary      │  │
│  │  RecordInput    │   │  PlotViewer              │  │
│  │  [Explain]      │   │  FeatureTable            │  │
│  └─────────────────┘   └──────────────────────────┘  │
└──────────────────────────────────────────────────────┘
           │  POST /api/explain
           ▼
┌──────────────────────────────────────────────────────┐
│              FastAPI + LangGraph Agent               │
│                                                      │
│  validate_and_load → predict                         │
│                          ├─▶ shap_tool  ─┐           │
│                          ├─▶ lime_tool  ─┼─▶ interpret (LLM)
│                          └─▶ pdp_tool   ─┘           │
└──────────────────────────────────────────────────────┘
```

---

## Demo



https://github.com/user-attachments/assets/c3c14ca1-4212-4d53-9637-50562285b2f3


## Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Python | 3.11+ | [python.org](https://www.python.org/downloads/) |
| [uv](https://docs.astral.sh/uv/) | latest | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Node.js | 18+ | [nodejs.org](https://nodejs.org/) |
| npm | 9+ | bundled with Node.js |

---

## Installation

### 1. Clone the repository

```bash
git clone <repo-url>
cd anomaly-explainer-agent
```

### 2. Install all dependencies

```bash
make install
```

This runs two steps in sequence:
- `cd backend && uv sync` — creates `.venv` and installs all Python packages from `uv.lock`
- `cd frontend && npm install` — installs all Node packages from `package-lock.json`

> To install manually without `make`:
> ```bash
> cd backend && uv sync
> cd ../frontend && npm install
> ```

### 3. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in your API key:

```dotenv
# Choose your LLM provider: "anthropic" or "gemini"
LLM_PROVIDER=anthropic

# Set the key for your chosen provider
ANTHROPIC_API_KEY=sk-ant-...        # get from console.anthropic.com
GEMINI_API_KEY=                     # get from aistudio.google.com

# Optional overrides (defaults shown)
PLOT_DIR=/tmp/anomaly-plots
SESSION_TTL=3600
```

> **No API key?** The agent still runs — SHAP/LIME/PDP plots are always generated and a rule-based text summary is used as fallback.

### 4. (Optional) Install Gemini support

```bash
cd backend && uv add --optional gemini langchain-google-genai
```

---

## Running the Application

You need **two terminals** — one for the backend, one for the frontend.

### Terminal 1 — Backend (FastAPI)

```bash
make dev-backend
```

Or manually:

```bash
cd backend
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API is now available at **http://localhost:8000**. You can explore the auto-generated docs at **http://localhost:8000/docs**.

### Terminal 2 — Frontend (Next.js)

```bash
make dev-frontend
```

Or manually:

```bash
cd frontend
npm run dev
```

Open **http://localhost:3000** in your browser.

---

## Using the UI

1. **Upload Model** — drag and drop a `.pkl` or `.joblib` scikit-learn anomaly detection model onto the drop zone
2. **Define Schema** — paste your feature schema as JSON, or click **Load example** for a template
3. **(Optional) Reference CSV** — upload a CSV of normal training data to improve SHAP/LIME background quality
4. Click **Upload & Register Model** — you'll see a green confirmation with the detected model type and feature count
5. **Input a Record** — fill in feature values using the auto-generated form, or toggle to **JSON paste** mode
6. Click **Explain** to run the pipeline
7. View results:
   - **Prediction badge** — anomaly / normal label with exact score
   - **AI Explanation** — LLM-generated (or rule-based) plain-English summary
   - **Plots** — SHAP waterfall, LIME bar chart, PDP curves (tabbed; click any plot to zoom)
   - **Feature table** — sortable table showing each feature's SHAP and LIME contribution

---

## Running Tests

```bash
make test
```

Or manually:

```bash
cd backend
uv run pytest
```

Expected output: **59 tests passed**.

---

## Demo Notebook

With the backend running, launch the end-to-end walkthrough notebook:

```bash
cd notebooks
jupyter notebook demo_walkthrough.ipynb
```

The notebook:
1. Trains an `IsolationForest` on synthetic credit-card transaction data
2. Saves the model and uploads it to the API
3. Explains a **normal** transaction (£55, near home)
4. Explains an **anomalous** transaction (£9 500, far from home, many transactions)
5. Displays all plots inline and shows a side-by-side score comparison

---

## API Reference

### `POST /api/upload-model`

Upload a serialised model and schema. Returns a `session_id` for subsequent explain calls.

**Form fields (multipart/form-data):**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model_file` | file | yes | `.pkl` or `.joblib` (max 50 MB) |
| `schema` | string (JSON) | yes | Feature schema (see format below) |
| `reference_csv` | file | no | Background data CSV for SHAP/LIME |

**Schema format:**
```json
{
  "features": {
    "amount":    {"type": "float", "min": 0.0, "max": 10000.0},
    "n_tx_24h":  {"type": "int",   "min": 0,   "max": 200},
    "category":  {"type": "categorical", "values": ["retail", "online", "atm"]}
  }
}
```

**Response:**
```json
{"session_id": "550e8400-...", "model_type": "IsolationForest", "feature_count": 4, "status": "ready"}
```

---

### `POST /api/explain`

Run the full SHAP + LIME + PDP + LLM pipeline for one input record.

**Request body (JSON):**
```json
{
  "session_id": "550e8400-...",
  "input_record": {"amount": 9500.0, "n_tx_24h": 35, "category": "online"}
}
```

**Response:**
```json
{
  "prediction": {
    "label": "anomaly",
    "anomaly_score": -0.2134,
    "model_type": "IsolationForest"
  },
  "explanations": {
    "shap_values":  {"amount": 0.1821, "n_tx_24h": 0.0943, "category": -0.0012},
    "lime_weights": {"amount > 5000": 0.182},
    "top_features": ["amount", "n_tx_24h"],
    "plots": {
      "shap_plot_url":       "/api/plot/abc123",
      "shap_force_plot_url": "/api/plot/abc124",
      "lime_plot_url":       "/api/plot/def456",
      "pdp_plot_urls":       ["/api/plot/ghi789", "/api/plot/ghi790"]
    }
  },
  "summary": {
    "text": "This transaction is anomalous primarily because the amount (£9 500) is far above typical values...",
    "feature_contributions": [
      {"feature": "amount", "impact": "high", "direction": "increases_anomaly", "reason": "..."}
    ]
  },
  "errors": []
}
```

---

### `GET /api/plot/{plot_id}`

Serve a generated plot as a PNG image (`Content-Type: image/png`).

---

## Supported Models

| Model | SHAP Explainer | Notes |
|-------|---------------|-------|
| `IsolationForest` | TreeExplainer | Fastest |
| `RandomForestClassifier` | TreeExplainer | |
| `ExtraTreesClassifier` | TreeExplainer | |
| `OneClassSVM` | KernelExplainer | Slower (~30 s) |
| `LocalOutlierFactor` | KernelExplainer | Requires `novelty=True` |
| Any sklearn-compatible | KernelExplainer | Automatic fallback |

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `anthropic` | `anthropic` or `gemini` |
| `ANTHROPIC_API_KEY` | _(empty)_ | Claude API key — get at [console.anthropic.com](https://console.anthropic.com) |
| `GEMINI_API_KEY` | _(empty)_ | Gemini API key — get at [aistudio.google.com](https://aistudio.google.com) |
| `PLOT_DIR` | `/tmp/anomaly-plots` | Directory where PNG plots are saved |
| `SESSION_TTL` | `3600` | Session expiry in seconds |

---

## Project Structure

```
anomaly-explainer-agent/
├── .env.example                    # Environment variable template
├── Makefile                        # dev-backend / dev-frontend / test / install
├── backend/
│   ├── pyproject.toml              # Python dependencies (managed by uv)
│   ├── uv.lock                     # Locked dependency versions
│   └── app/
│       ├── main.py                 # FastAPI app factory + lifespan
│       ├── config.py               # Settings from environment
│       ├── routers/
│       │   ├── upload.py           # POST /api/upload-model, GET /api/plot/{id}
│       │   └── explain.py          # POST /api/explain
│       ├── agent/
│       │   ├── graph.py            # LangGraph StateGraph definition
│       │   ├── state.py            # ExplainerState TypedDict
│       │   └── nodes/
│       │       ├── predict.py      # Prediction node
│       │       ├── shap_tool.py    # SHAP waterfall + bar chart
│       │       ├── lime_tool.py    # LIME bar chart
│       │       ├── pdp_tool.py     # Partial dependence plots
│       │       └── interpret.py    # LLM interpretation node
│       └── services/
│           ├── model_loader.py     # Load .pkl / .joblib, detect model type
│           ├── schema_validator.py # Validate input records against schema
│           ├── plot_generator.py   # PNG save utilities + background data generation
│           └── llm_client.py       # Anthropic / Gemini client wrapper
├── frontend/
│   ├── package.json
│   └── src/
│       ├── app/
│       │   ├── page.tsx            # Main two-column page
│       │   ├── layout.tsx          # Root layout + metadata
│       │   └── globals.css         # Tailwind CSS v4 base styles
│       ├── components/
│       │   ├── ModelUpload.tsx     # Drag-drop model + schema editor + ref CSV
│       │   ├── SchemaEditor.tsx    # Schema feature table preview
│       │   ├── RecordInput.tsx     # Auto-generated form + JSON-paste toggle
│       │   ├── PredictionBadge.tsx # Anomaly / normal result banner
│       │   ├── PlotViewer.tsx      # SHAP / LIME / PDP tabbed viewer + lightbox
│       │   ├── FeatureTable.tsx    # Sortable feature contribution table
│       │   ├── ExplanationSummary.tsx # LLM summary + feature contributions
│       │   └── LoadingState.tsx    # Step indicator + skeleton loaders
│       ├── api/client.ts           # uploadModel() / explainRecord() fetch wrappers
│       ├── context/AppContext.tsx  # useReducer global state
│       └── types/index.ts          # TypeScript type definitions
└── notebooks/
    └── demo_walkthrough.ipynb      # End-to-end demo with synthetic data
```
