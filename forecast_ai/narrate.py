"""
Turn a backtest table into a short plain-English summary for a non-technical reader.
Uses any OpenAI-compatible API if LLM_API_KEY / OPENAI_API_KEY is set; otherwise a
deterministic template ("mock") writes it, so reports always work offline.
"""
from __future__ import annotations

import json
import os
import urllib.request

import pandas as pd

from .tables import markdown_table


def template_summary(table: pd.DataFrame, dataset: str, horizon: int) -> str:
    best = table.index[0]
    base = table.loc["Seasonal naive"] if "Seasonal naive" in table.index else table.iloc[-1]
    b = table.loc[best]
    gain = (base["MAE"] - b["MAE"]) / base["MAE"] * 100
    direction = "over" if b["Bias"] > 0 else "under"
    caveat = ("so treat the forecast as close to unbiased." if abs(b["Bias"]) < 2
              else "so consider adjusting stock or staffing plans accordingly.")
    lines = [
        f"**{best}** is the most accurate model for {dataset} over a {horizon}-day horizon, "
        f"with an average error of **{b['MAPE']:.1f}%** (MASE {b['MASE']:.2f}).",
        f"That is **{gain:.0f}% lower error** than the seasonal-naive baseline (\"same day last week\", "
        f"{base['MAPE']:.1f}% error).",
        f"On average it {direction}-forecasts by {abs(b['Bias']):.1f}%, {caveat}",
    ]
    if "Holt-Winters" in table.index and best != "Holt-Winters":
        lines.append(f"Holt-Winters ({table.loc['Holt-Winters', 'MAPE']:.1f}%) is a fast, transparent fallback "
                     "that needs no feature engineering.")
    return " ".join(lines)


def llm_summary(table: pd.DataFrame, dataset: str, horizon: int, env=None, opener=urllib.request.urlopen) -> tuple[str, str]:
    """Return (source, summary). Falls back to the template on missing key or any error."""
    env = dict(os.environ) if env is None else env
    key = env.get("LLM_API_KEY") or env.get("OPENAI_API_KEY")
    if not key:
        return "template", template_summary(table, dataset, horizon)
    base = (env.get("LLM_BASE_URL") or env.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
    model = env.get("LLM_MODEL") or env.get("OPENAI_MODEL") or "gpt-4o-mini"
    prompt = (f"Backtest results for forecasting daily {dataset} {horizon} days ahead "
              f"(lower MAPE/MASE is better; Bias % > 0 means over-forecasting):\n{markdown_table(table)}\n\n"
              "Write 3-4 sentences for a retail manager: which model to use, how accurate it is versus the "
              "baseline, and one practical caveat. Use Markdown bold for the key numbers.")
    body = {"model": model, "temperature": 0.3, "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(f"{base}/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    try:
        with opener(req, timeout=60) as resp:
            text = json.loads(resp.read().decode())["choices"][0]["message"]["content"].strip()
        return model, text
    except Exception as err:  # network, auth, schema: never fail the report
        return f"template (LLM failed: {err.__class__.__name__})", template_summary(table, dataset, horizon)
