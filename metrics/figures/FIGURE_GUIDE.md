# Milestone 5B — Benchmark Figures

Generate SVGs from the three committed JSON datasets:

```bash
python figures/generate_figures.py
```

No GPU, Modal deployment, additional Python dependencies, or network access are required for chart generation. The script expects the original experiment filenames in `results/` and writes SVGs beside itself.

| Figure | Source dataset | Definition |
|---|---|---|
| `01_baseline_throughput.svg` | `20261019_014825.json` | Arithmetic mean of batch output-token throughput by workload and concurrency |
| `02_baseline_ttft.svg` | `20261019_014825.json` | Median client-observed TTFT pooled across successful requests by workload and concurrency |
| `03_budget_ttft_delta.svg` | `20261019_024946.json`, `20261019_030158.json` | Percent change in pooled TTFT p50/p95, `(4096-2048)/2048*100` |
| `04_long_ttft_ranked.svg` | Both 20-trial files | Within each long-prompt concurrency-8 batch, sort eight TTFTs; plot the median and interquartile range for each rank across 20 trials |

For GitHub READMEs, embed the SVG files rather than the PNGs, for crisp resizing:

```markdown
![Aggregate output throughput by concurrency](figures/01_baseline_throughput.svg)
![Median TTFT by concurrency](figures/02_baseline_ttft.svg)
![Scheduling-budget effect on TTFT](figures/03_budget_ttft_delta.svg)
![Long-prompt ranked TTFT distribution](figures/04_long_ttft_ranked.svg)
```

## Interpretation cautions

- All latency is observed by the client; it is not GPU-only time.
- The five-batch initial baseline has limited statistical power; p95 is sensitive to individual requests.
- The 20-trial budget comparison always tested 2048 before 4096; deltas are descriptive and not causal proof.
- Rank order in figure 4 is *within-batch arrival order*, not fixed client request ID or a traced scheduler order.
- Reproducible stair-step appearance suggests staggered first-token arrivals, not a proven chunked-prefill execution trace.
- The original `generated_text` is retained in input JSON but is not needed to produce any figure.
