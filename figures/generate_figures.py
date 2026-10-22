from __future__ import annotations

import argparse
import json
import math
import statistics
from html import escape
from pathlib import Path

FILES = {
    "baseline": "20261019_014825.json",
    "control": "20261019_024946.json",
    "candidate": "20261019_030158.json",
}
BLUE = "#0072B2"
ORANGE = "#C55A11"
INK = "#17212b"
WORKLOADS = (
    ("short", "circle", BLUE, -10, "Short · 256 input tokens"),
    ("long", "diamond", ORANGE, 10, "Long · 2048 input tokens"),
)

def read_batches(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError(f"Expected nonempty benchmark list: {path}")
    return data

def groups(data: list[dict], workload: str, concurrency: int) -> list[dict]:
    batches = [
        b for b in data if b["workload"] == workload and b["concurrency"] == concurrency
    ]
    if not batches:
        raise ValueError(f"Missing {workload=}, {concurrency=}")
    for batch in batches:
        requests = batch["results"]
        if (
            batch.get("failed_requests", 0)
            or len(requests) != concurrency
            or not all(r["success"] for r in requests)
        ):
            raise ValueError(
                f"Incomplete {workload=}, {concurrency=}, trial={batch.get('trial')}"
            )
    return batches

def get_ttfts(batches: list[dict]) -> list[float]:
    values = [r["ttft_ms"] for b in batches for r in b["results"]]
    if not values or any(v is None or not math.isfinite(v) or v < 0 for v in values):
        raise ValueError(
            "Expected finite nonnegative TTFT for every successful request"
        )
    return values

def pct(values: list[float], percentile: int) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("No samples")
    index = (len(ordered) - 1) * percentile / 100
    low = math.floor(index)
    fraction = index - low
    return ordered[low] + fraction * (
        ordered[min(low + 1, len(ordered) - 1)] - ordered[low]
    )

def ranked_summary(data: list[dict]) -> tuple[list[float], list[float], list[float]]:
    trials = [sorted(get_ttfts([batch])) for batch in groups(data, "long", 8)]
    return tuple(
        [pct([trial[i] for trial in trials], p) for i in range(8)] for p in (50, 25, 75)
    )

class SVG:
    W, H = 960, 610
    L, R, T, B = 104, 904, 160, 500

    def __init__(self, title: str, subtitle: str, footer: str, note: str = ""):
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.W}" height="{self.H}" '
            f'viewBox="0 0 {self.W} {self.H}" role="img" aria-label="{escape(title, quote=True)}">',
            f"<title>{escape(title)}</title>",
            f"<desc>{escape(subtitle + '. ' + footer + '. ' + note)}</desc>",
            '<rect width="100%" height="100%" fill="#ffffff"/>',
            "<style>"
            f"text{{font-family:Arial,Helvetica,sans-serif;fill:{INK}}}"
            ".title{font-size:25px;font-weight:700}"
            ".subtitle{font-size:14px;opacity:.72}"
            ".footnote{font-size:12px;opacity:.66}"
            ".tick{font-size:13px;opacity:.83}"
            ".axislabel{font-size:15px;font-weight:600}"
            ".legend{font-size:14px}"
            ".value{font-size:14px;font-weight:600}"
            "</style>",
        ]
        self.text(44, 44, title, "title")
        self.text(44, 72, subtitle, "subtitle")
        self.text(44, 575, footer, "footnote")
        # Methodological detail remains in the accessible description.
        self.marker_layers = []

    def raw(self, markup: str):
        self.parts.append(markup)

    def metadata(self, values: dict):
        self.raw(f"<metadata>{escape(json.dumps(values, sort_keys=True))}</metadata>")

    def text(self, x, y, value, css="", anchor=None, rotate=None):
        attrs = f' x="{x:.2f}" y="{y:.2f}" class="{css}"'
        if anchor:
            attrs += f' text-anchor="{anchor}"'
        if rotate is not None:
            attrs += f' transform="rotate({rotate} {x:.2f} {y:.2f})"'
        self.raw(f"<text{attrs}>{escape(str(value))}</text>")

    def line(self, x1, y1, x2, y2, color=INK, width=1.5, opacity=1):
        self.raw(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{color}" stroke-width="{width}" opacity="{opacity}"/>'
        )

    def point(self, x, y, shape, color):
        if shape == "circle":
            self.raw(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5.3" fill="{color}" '
                'stroke="white" stroke-width="1.4"/>'
            )
        else:
            d = 6.2
            self.raw(
                f'<path d="M {x:.2f},{y - d:.2f} L {x + d:.2f},{y:.2f} '
                f'L {x:.2f},{y + d:.2f} L {x - d:.2f},{y:.2f} Z" '
                f'fill="{color}" stroke="white" stroke-width="1.4"/>'
            )

    def whisker(self, x, low, high, color):
        y1, y2 = self.map_y(low), self.map_y(high)
        self.line(x, y1, x, y2, color, 1.9)
        self.line(x - 4, y1, x + 4, y1, color, 1.9)
        self.line(x - 4, y2, x + 4, y2, color, 1.9)

    def axes(
        self,
        xlabel,
        ylabel,
        min_y,
        max_y,
        ticks,
        xlabels,
        tick_fmt=lambda value: f"{value:g}",
    ):
        self.min_y, self.max_y = min_y, max_y
        for value in ticks:
            y = self.map_y(value)
            self.line(self.L, y, self.R, y, width=1, opacity=0.12)
            self.text(self.L - 16, y + 5, tick_fmt(value), "tick", "end")
        self.line(self.L, self.T, self.L, self.B, opacity=0.65)
        self.line(self.L, self.B, self.R, self.B, opacity=0.65)
        for x, label in xlabels:
            self.text(x, self.B + 28, label, "tick", "middle")
        self.text((self.L + self.R) / 2, 555, xlabel, "axislabel", "middle")
        self.text(26, (self.T + self.B) / 2, ylabel, "axislabel", "middle", -90)

    def map_y(self, value):
        return self.B - (value - self.min_y) / (self.max_y - self.min_y) * (
            self.B - self.T
        )

    def map_x_number(self, value, minimum, maximum):
        return self.L + (value - minimum) / (maximum - minimum) * (self.R - self.L)

    def legend(self, entries, whiskers=True):
        # Keep both series markers above every connecting line.
        for xs, medians, shape, color, offset in self.marker_layers:
            for x, value in zip(xs, medians):
                self.point(x + offset, self.map_y(value), shape, color)
        self.marker_layers.clear()
        for index, (label, shape, color) in enumerate(entries):
            x = 116 + index * 300
            if whiskers:
                self.line(x - 16, 112, x + 16, 112, color, 2)
            if shape == "bar":
                self.raw(
                    f'<rect x="{x - 6}" y="106" width="12" height="12" fill="{color}"/>'
                )
            else:
                self.point(x, 112, shape, color)
            self.text(x + (27 if whiskers else 16), 117, label, "legend")

    def paired_intervals(self, xs, medians, lows, highs, shape, color, offset):
        for x, low, high in zip(xs, lows, highs):
            self.whisker(x + offset, low, high, color)
        points = " ".join(
            f"{x + offset:.2f},{self.map_y(value):.2f}" for x, value in zip(xs, medians)
        )
        self.raw(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2" '
            'stroke-linecap="round" stroke-linejoin="round"/>'
        )
        self.marker_layers.append((xs, medians, shape, color, offset))

    def save(self, path: Path):
        path.write_text("\n".join(self.parts + ["</svg>", ""]), encoding="utf-8")
        print("Saved", path)

def baseline_throughput(data, out):
    g = SVG(
        "Aggregate output throughput scales with concurrency",
        "Initial baseline · Qwen3-4B on NVIDIA L4 · vLLM 0.21.0",
        "Markers = mean batch throughput · Whiskers = minimum and maximum across 5 batches",
        "Paired offsets separate workloads at each concurrency · "
        "whiskers are descriptive ranges, not confidence intervals",
    )
    concurrency = [1, 4, 8]
    xs = [g.map_x_number(c, 0.5, 8.5) for c in concurrency]
    g.axes(
        "Concurrent requests",
        "Output tokens / second",
        0,
        225,
        [0, 50, 100, 150, 200],
        list(zip(xs, map(str, concurrency))),
    )
    summaries = {}
    for name, shape, color, offset, label in WORKLOADS:
        rates = [
            [b["output_tokens_per_sec"] for b in groups(data, name, c)]
            for c in concurrency
        ]
        means = [statistics.mean(values) for values in rates]
        lows, highs = (
            [min(values) for values in rates],
            [max(values) for values in rates],
        )
        g.paired_intervals(xs, means, lows, highs, shape, color, offset)
        for x, value, low, high in zip(xs, means, lows, highs):
            y = g.map_y(high) - 14 if name == "short" else g.map_y(low) + 22
            g.text(x + offset, y, f"{value:.1f}", "value", "middle")
        summaries[name] = {
            "mean": means,
            "min": lows,
            "max": highs,
        }
    g.legend([(label, shape, color) for _, shape, color, _, label in WORKLOADS])
    g.metadata(
        {
            "source": FILES["baseline"],
            "concurrency": concurrency,
            "statistics": summaries,
        }
    )
    g.save(out / "01_baseline_throughput.svg")

def baseline_ttft(data, out):
    g = SVG(
        "Median first-token latency rises with concurrency",
        "Initial baseline · client-observed TTFT · cold starts excluded",
        "Markers = median TTFT · Whiskers = interquartile range "
        "of requests pooled across 5 batches",
        "Paired offsets separate workloads at each concurrency · "
        "IQR describes request spread, not confidence intervals",
    )
    concurrency = [1, 4, 8]
    xs = [g.map_x_number(c, 0.5, 8.5) for c in concurrency]
    g.axes(
        "Concurrent requests",
        "TTFT (ms)",
        0,
        3400,
        [0, 500, 1000, 1500, 2000, 2500, 3000],
        list(zip(xs, map(str, concurrency))),
    )
    summaries = {}
    for name, shape, color, offset, label in WORKLOADS:
        requests = [get_ttfts(groups(data, name, c)) for c in concurrency]
        medians, lows, highs = (
            [pct(values, p) for values in requests] for p in (50, 25, 75)
        )
        g.paired_intervals(xs, medians, lows, highs, shape, color, offset)
        for x, value, low, high in zip(xs, medians, lows, highs):
            y = g.map_y(low) + 22 if name == "short" else g.map_y(high) - 14
            g.text(x + offset, y, f"{value:.0f}", "value", "middle")
        summaries[name] = {
            "median": medians,
            "p25": lows,
            "p75": highs,
        }
    g.legend([(label, shape, color) for _, shape, color, _, label in WORKLOADS])
    g.metadata(
        {
            "source": FILES["baseline"],
            "concurrency": concurrency,
            "statistics": summaries,
        }
    )
    g.save(out / "02_baseline_ttft.svg")

def budget_effect(control, candidate, out):
    labels, changes = [], []
    for workload in ("short", "long"):
        for percentile in (50, 95):
            old = pct(get_ttfts(groups(control, workload, 8)), percentile)
            new = pct(get_ttfts(groups(candidate, workload, 8)), percentile)
            labels.append(f"{workload.capitalize()} p{percentile}")
            changes.append(100 * (new - old) / old)
    g = SVG(
        "Larger scheduling budget did not improve TTFT",
        "Relative change when raising max_num_batched_tokens from 2048 to 4096",
        "Bars = percentage change in TTFT · Positive values = worse performance",
        "20 measured batches × 8 requests per workload/config; fixed run order; "
        "descriptive comparison without confidence intervals.",
    )
    xs = [g.L + 130 + i * 180 for i in range(4)]
    g.axes(
        "Workload / TTFT percentile",
        "Change vs. 2048 (%)",
        0,
        16,
        [0, 4, 8, 12, 16],
        list(zip(xs, labels)),
        lambda y: f"+{y:g}%" if y else "0%",
    )
    for index, (x, value) in enumerate(zip(xs, changes)):
        y = g.map_y(value)
        color = BLUE if index < 2 else ORANGE
        g.raw(
            f'<rect x="{x - 30:.2f}" y="{y:.2f}" width="60" height="{g.B - y:.2f}" fill="{color}"/>'
        )
        g.text(x, y - 13, f"{value:+.2f}%", "value", "middle")
    g.legend(
        [
            ("Short · 256 input tokens", "bar", BLUE),
            ("Long · 2048 input tokens", "bar", ORANGE),
        ],
        whiskers=False,
    )
    g.metadata(
        {
            "sources": [FILES["control"], FILES["candidate"]],
            "labels": labels,
            "change_pct": changes,
        }
    )
    g.save(out / "03_budget_ttft_delta.svg")

def ranked_ttft(control, candidate, out):
    g = SVG(
        "Long-prompt TTFT by arrival rank",
        "Concurrency 8 · 2048-token inputs · 20 batches per scheduler budget",
        "Markers = median TTFT at each rank · Whiskers = interquartile range across 20 batches",
        "Paired offsets separate budgets at each rank · "
        "sorted ranks do not preserve request identity",
    )
    xs = [g.map_x_number(rank, 0.5, 8.5) for rank in range(1, 9)]
    g.axes(
        "First-token arrival rank (fastest to slowest)",
        "TTFT (ms)",
        0,
        3400,
        [0, 500, 1000, 1500, 2000, 2500, 3000],
        list(zip(xs, map(str, range(1, 9)))),
    )
    summaries = {}
    for name, data, shape, color, offset in (
        ("control", control, "circle", BLUE, -10),
        ("candidate", candidate, "diamond", ORANGE, 10),
    ):
        medians, lows, highs = ranked_summary(data)
        g.paired_intervals(xs, medians, lows, highs, shape, color, offset)
        summaries[name] = {
            "median": medians,
            "p25": lows,
            "p75": highs,
        }
    g.legend(
        [
            ("2048-token budget", "circle", BLUE),
            ("4096-token budget", "diamond", ORANGE),
        ]
    )
    g.metadata(
        {
            "sources": [FILES["control"], FILES["candidate"]],
            "statistics": summaries,
        }
    )
    g.save(out / "04_long_ttft_ranked.svg")

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "results",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parent
    )
    args = parser.parse_args()
    # Read inputs before creating output files, so missing datasets fail cleanly.
    data = {
        name: read_batches(args.results_dir / filename)
        for name, filename in FILES.items()
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline_throughput(data["baseline"], args.output_dir)
    baseline_ttft(data["baseline"], args.output_dir)
    budget_effect(data["control"], data["candidate"], args.output_dir)
    ranked_ttft(data["control"], data["candidate"], args.output_dir)

if __name__ == "__main__":
    main()
