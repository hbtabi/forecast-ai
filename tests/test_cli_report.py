import json

import pandas as pd

from forecast_ai.__main__ import future_frame, main
from forecast_ai.narrate import llm_summary, template_summary
from forecast_ai.tables import markdown_table, text_table

TABLE = pd.DataFrame(
    {"MAE": [100.0, 150.0, 200.0], "RMSE": [120.0, 180.0, 260.0], "MAPE": [6.5, 8.6, 11.9], "MASE": [0.7, 0.96, 1.28],
     "Bias": [2.0, -1.9, 1.7], "fit_s": [1.0, 0.1, 0.0]},
    index=pd.Index(["Gradient boosting", "Holt-Winters", "Seasonal naive"], name="model"),
)


def test_tables_render():
    md = markdown_table(TABLE, bold_first=True)
    assert md.splitlines()[2].startswith("| **Gradient boosting** | **100.0**")
    assert "6.50%" in md and "+2.00%" in md
    assert "Seasonal naive" in text_table(TABLE)


def test_template_summary_mentions_winner_and_gain():
    s = template_summary(TABLE, "sales", 28)
    assert "**Gradient boosting**" in s and "50% lower error" in s and "over-forecasts" in s


def test_llm_summary_offline_and_with_api():
    assert llm_summary(TABLE, "sales", 28, env={})[0] == "template"

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": "Use **GBM**."}}]}).encode()

    seen = {}

    def opener(req, timeout):
        seen["url"], seen["body"] = req.full_url, json.loads(req.data)
        return Resp()

    env = {"OPENAI_API_KEY": "k", "OPENAI_BASE_URL": "http://localhost:1234/v1", "OPENAI_MODEL": "qwen"}
    assert llm_summary(TABLE, "sales", 28, env=env, opener=opener) == ("qwen", "Use **GBM**.")
    assert seen["url"] == "http://localhost:1234/v1/chat/completions"
    assert "| Gradient boosting |" in seen["body"]["messages"][0]["content"]

    def broken(req, timeout):
        raise OSError("offline")

    src, text = llm_summary(TABLE, "sales", 28, env=env, opener=broken)
    assert src.startswith("template (LLM failed") and "Gradient boosting" in text


def test_future_frame_marks_holidays_and_promos():
    df = pd.DataFrame({"date": pd.date_range("2026-12-20", periods=5), "value": 1.0})
    fut = future_frame(df, 7, ["2026-12-27"])
    assert fut["date"].iloc[0] == pd.Timestamp("2026-12-25")
    assert fut["holiday"].tolist()[:2] == [1, 1]
    assert fut.loc[fut["date"] == "2026-12-27", "promo"].item() == 1


def test_cli_end_to_end(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    csv = tmp_path / "shop.csv"
    from forecast_ai.data import synthetic_series

    synthetic_series("footfall", days=520).to_csv(csv, index=False)
    report = tmp_path / "out" / "report.md"
    assert main(["compare", "--dataset", str(csv), "--horizon", "14", "--folds", "2",
                 "--charts-dir", str(tmp_path / "charts"), "--report", str(report)]) == 0
    text = report.read_text()
    assert "# Forecast report: shop" in text and "![Holdout forecast](../charts/shop_forecast.png)" in text
    assert (tmp_path / "charts" / "shop_backtest.png").exists()

    out_csv = tmp_path / "f.csv"
    assert main(["forecast", "--dataset", str(csv), "--horizon", "7", "--model", "hw", "--out", str(out_csv),
                 "--chart", str(tmp_path / "f.png")]) == 0
    f = pd.read_csv(out_csv)
    assert len(f) == 7 and (f["lower_80"] <= f["forecast"]).all() and (f["forecast"] <= f["upper_80"]).all()
    assert "saved chart" in capsys.readouterr().out
