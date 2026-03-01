"""LangGraph node: LLM interpretation of SHAP / LIME / PDP results.

Reads explainability outputs from ExplainerState, encodes available plot
images as base64, constructs a structured prompt, calls the configured LLM,
and parses the JSON response into ``explanation_summary`` and
``feature_contributions``.

Falls back to a rule-based summary when:
- No LLM API key is configured.
- The LLM call fails after all retries.
- The LLM response cannot be parsed as valid JSON.
"""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Any

from app.agent.state import ExplainerState
from app.config import ANTHROPIC_API_KEY, GEMINI_API_KEY, LLM_PROVIDER
from app.services.llm_client import LLMError, call_llm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _encode_image(path: str | None) -> dict | None:
    """Read a PNG file and return a base64 image dict, or None if unavailable."""
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        logger.debug("Plot file not found, skipping: %s", path)
        return None
    try:
        data = base64.b64encode(p.read_bytes()).decode("utf-8")
        return {"data": data, "media_type": "image/png"}
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to encode image %s: %s", path, exc)
        return None


def _collect_images(state: ExplainerState) -> list[dict]:
    """Gather all available plot images as base64 dicts."""
    candidates: list[str | None] = [
        state.get("shap_plot_path"),
        state.get("shap_force_plot_path"),
        state.get("lime_plot_path"),
        *(state.get("pdp_plot_paths") or []),
    ]
    images: list[dict] = []
    for path in candidates:
        img = _encode_image(path)
        if img:
            images.append(img)
    return images


def _format_dict(d: dict | None, limit: int = 10) -> str:
    """Format a dict as aligned key=value lines, sorted by absolute value."""
    if not d:
        return "  (none available)"
    items = sorted(d.items(), key=lambda kv: abs(kv[1]), reverse=True)[:limit]
    lines = [f"  {k}: {v:+.4f}" for k, v in items]
    return "\n".join(lines)


def _build_prompt(state: ExplainerState, n_pdp_plots: int) -> str:
    """Construct the structured prompt for the LLM."""
    model_type = state.get("model_type") or "unknown"
    prediction_label = state.get("prediction_label") or "unknown"
    anomaly_score = state.get("anomaly_score") or 0.0
    shap_values = state.get("shap_values") or {}
    lime_weights = state.get("lime_weights") or {}

    pdp_note = f", {n_pdp_plots} PDP plot(s)" if n_pdp_plots else ""

    shap_section = _format_dict(shap_values)
    lime_section = _format_dict(lime_weights)

    prompt = f"""\
You are an ML explainability expert specializing in anomaly detection.

Model type: {model_type}
Prediction: {prediction_label} (anomaly score: {anomaly_score:.4f})

SHAP feature attributions (higher absolute value = more influential):
{shap_section}

LIME feature weights:
{lime_section}

Attached plots: SHAP waterfall, SHAP impact chart{pdp_note}

Respond with ONLY valid JSON (no markdown fences, no explanation outside the JSON):
{{
  "summary": "Plain-English explanation (2-3 sentences)",
  "top_contributors": [
    {{"feature": "...", "impact": "high|medium|low", "direction": "increases_anomaly|decreases_anomaly", "reason": "..."}}
  ],
  "consistency_note": "Agreement/disagreement between SHAP and LIME",
  "next_steps": ["action 1", "action 2"]
}}"""
    return prompt


# ---------------------------------------------------------------------------
# Rule-based fallback
# ---------------------------------------------------------------------------


def _rule_based_summary(state: ExplainerState) -> dict:
    """Generate a deterministic text summary from SHAP values without an LLM."""
    prediction_label = state.get("prediction_label") or "unknown"
    anomaly_score = state.get("anomaly_score") or 0.0
    shap_values: dict = state.get("shap_values") or {}

    if not shap_values:
        summary = (
            f"The prediction '{prediction_label}' (score {anomaly_score:.4f}) "
            "could not be further explained: no SHAP values available."
        )
        return {"explanation_summary": summary, "feature_contributions": []}

    sorted_features = sorted(
        shap_values.items(), key=lambda kv: abs(kv[1]), reverse=True
    )

    top3 = sorted_features[:3]
    parts = []
    for feat, val in top3:
        direction = "increases" if val > 0 else "decreases"
        parts.append(f"{feat} ({val:+.4f}, {direction} anomaly score)")

    if parts:
        summary = (
            f"The prediction '{prediction_label}' (score {anomaly_score:.4f}) "
            f"was most influenced by {parts[0]}"
        )
        if len(parts) > 1:
            summary += ", followed by " + ", ".join(parts[1:])
        summary += "."
    else:
        summary = (
            f"The prediction '{prediction_label}' (score {anomaly_score:.4f}) "
            "had no dominant features."
        )

    def _impact(val: float) -> str:
        av = abs(val)
        if av >= 0.2:
            return "high"
        if av >= 0.05:
            return "medium"
        return "low"

    feature_contributions = [
        {
            "feature": feat,
            "impact": _impact(val),
            "direction": "increases_anomaly" if val > 0 else "decreases_anomaly",
        }
        for feat, val in top3
    ]

    return {
        "explanation_summary": summary,
        "feature_contributions": feature_contributions,
    }


# ---------------------------------------------------------------------------
# API-key guard
# ---------------------------------------------------------------------------


def _has_api_key() -> bool:
    """Return True when the configured provider has a non-empty API key."""
    provider = LLM_PROVIDER.lower()
    if provider == "anthropic":
        return bool(ANTHROPIC_API_KEY)
    if provider == "gemini":
        return bool(GEMINI_API_KEY)
    return False


# ---------------------------------------------------------------------------
# LLM response parsing
# ---------------------------------------------------------------------------


def _parse_llm_response(raw: str) -> dict:
    """Parse the LLM JSON response; fall back gracefully on parse errors."""
    # Strip accidental markdown code fences if the LLM disobeyed instructions
    stripped = raw.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        inner_lines: list[str] = []
        in_block = False
        for line in lines:
            if line.startswith("```") and not in_block:
                in_block = True
                continue
            if line.startswith("```") and in_block:
                break
            if in_block:
                inner_lines.append(line)
        stripped = "\n".join(inner_lines)

    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        logger.warning("LLM response was not valid JSON; using raw text as summary.")
        return {
            "explanation_summary": raw,
            "feature_contributions": [],
        }

    summary = data.get("summary", "")
    top_contributors = data.get("top_contributors", [])

    # Normalise top_contributors into feature_contributions schema
    feature_contributions: list[dict] = []
    for item in top_contributors:
        if isinstance(item, dict):
            feature_contributions.append(
                {
                    "feature": item.get("feature", ""),
                    "impact": item.get("impact", ""),
                    "direction": item.get("direction", ""),
                    "reason": item.get("reason", ""),
                }
            )

    result: dict[str, Any] = {
        "explanation_summary": summary,
        "feature_contributions": feature_contributions,
    }
    # Preserve extra fields from the LLM response
    if "consistency_note" in data:
        result["consistency_note"] = data["consistency_note"]
    if "next_steps" in data:
        result["next_steps"] = data["next_steps"]

    return result


# ---------------------------------------------------------------------------
# Public node
# ---------------------------------------------------------------------------


async def interpret_node(state: ExplainerState) -> dict:
    """LangGraph node: call LLM to interpret SHAP / LIME / PDP results.

    Reads from state:
        model_type, prediction_label, anomaly_score,
        shap_values, lime_weights,
        shap_plot_path, shap_force_plot_path, lime_plot_path, pdp_plot_paths

    Writes to state:
        explanation_summary, feature_contributions
        (plus optional consistency_note, next_steps)
    """
    errors: list[str] = list(state.get("errors", []))

    # Collect and encode available plot images
    images = _collect_images(state)
    n_pdp_plots = len(state.get("pdp_plot_paths") or [])

    # If no API key is configured, skip the LLM and use the rule-based fallback
    if not _has_api_key():
        logger.info(
            "No LLM API key configured for provider '%s'; using rule-based summary.",
            LLM_PROVIDER,
        )
        result = _rule_based_summary(state)
        result["errors"] = errors
        return result

    # Build prompt
    prompt = _build_prompt(state, n_pdp_plots)

    # Call LLM with retry
    try:
        raw_response = await call_llm(prompt, images if images else None)
    except LLMError as exc:
        logger.error("LLM interpretation failed: %s. Falling back to rule-based.", exc)
        errors.append(f"interpret_llm_error: {exc}")
        result = _rule_based_summary(state)
        result["errors"] = errors
        return result

    # Parse response
    result = _parse_llm_response(raw_response)
    result["errors"] = errors
    return result
