"""Small helpers to render DataFrames as Markdown or plain text without extra dependencies."""
from __future__ import annotations

import pandas as pd

FORMATS = {"MAE": "{:,.1f}", "RMSE": "{:,.1f}", "MAPE": "{:.2f}%", "MASE": "{:.3f}", "Bias": "{:+.2f}%", "fit_s": "{:.2f}s"}


def _fmt(col: str, v) -> str:
    return FORMATS.get(col, "{}").format(v) if isinstance(v, (int, float)) else str(v)


def markdown_table(df: pd.DataFrame, bold_first: bool = False) -> str:
    cols = [df.index.name or ""] + list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] + ["---:"] * len(df.columns)) + "|"]
    for i, (idx, row) in enumerate(df.iterrows()):
        cells = [str(idx)] + [_fmt(c, row[c]) for c in df.columns]
        if bold_first and i == 0:
            cells = [f"**{c}**" for c in cells]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def text_table(df: pd.DataFrame) -> str:
    cols = [df.index.name or ""] + list(df.columns)
    rows = [[str(i)] + [_fmt(c, r[c]) for c in df.columns] for i, r in df.iterrows()]
    widths = [max(len(x) for x in col) for col in zip(cols, *rows, strict=False)]
    def fmt(cells: list[str]) -> str:
        pairs = enumerate(zip(cells, widths, strict=False))
        return "  ".join(c.ljust(w) if j == 0 else c.rjust(w) for j, (c, w) in pairs)
    return "\n".join([fmt(cols), "  ".join("─" * w for w in widths)] + [fmt(r) for r in rows])
