"""Rebuild README.md and the charts in assets/ from the saved results in results/."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from instaquery import report  # noqa: E402
from instaquery.config import Config  # noqa: E402

R, A = ROOT / "results", ROOT / "assets"
A.mkdir(exist_ok=True)
metrics = report.load_metrics(R)
ts = json.loads((R / "train_stats.json").read_text())
bench = json.loads((R / "benchmark.json").read_text())
em = json.loads((R / "official_exact_match.json").read_text())
loss = json.loads((R / "training_loss.json").read_text())
report.chart_ex_by_variant(metrics, A / "ex_by_variant.png")
report.chart_by_difficulty(metrics, A / "ex_by_difficulty.png")
report.chart_training_loss([{"step": s, "loss": l} for s, l in loss], A / "training_loss.png")
report.chart_throughput(bench, A / "throughput.png")
readme = report.build_readme((ROOT / "templates" / "README.md.tmpl").read_text(), Config(run_mode="full"), metrics, ts,
                             bench, (R / "error_analysis.md").read_text(), em,
                             (ROOT / "docs" / "findings.md").read_text(),
                             (ROOT / "docs" / "engineering_notes.md").read_text())
(ROOT / "README.md").write_text(readme)
print("README.md and charts rebuilt")
