"""Generate four reproducible SVG figures for the LLM Inference Optimization Lab.

Run from the project root:
    python figures/generate_figures.py

Uses only Python's standard library. Requires the three committed benchmark
JSON files in results/. Does not contact Modal or require a GPU.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from html import escape
from pathlib import Path

FILES = {"baseline": "20261019_014825.json", "control": "20261019_024946.json", "candidate": "20261019_030158.json"}


def read_batches(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError(f"Expected nonempty benchmark list: {path}")
    return data


def groups(data: list[dict], workload: str, concurrent: int) -> list[dict]:
    group = [b for b in data if b["workload"] == workload and b["concurrency"] == concurrent]
    if not group or any(b.get("failed_requests", 0) for b in group):
        raise ValueError(f"Missing or incomplete {workload=} {concurrent=}")
    return group


def get_ttfts(group: list[dict]) -> list[float]:
    return [r["ttft_ms"] for b in group for r in b["results"] if r["success"] and r["ttft_ms"] is not None]


def pct(values: list[float], percentile: int) -> float:
    """Linear interpolation, matching numpy.percentile's default method."""
    values = sorted(values)
    if not values:
        raise ValueError("No samples")
    index = (len(values) - 1) * percentile / 100
    i, frac = math.floor(index), index % 1
    return values[i] + frac * (values[min(i + 1, len(values) - 1)] - values[i])


class SVG:
    W, H = 960, 610
    L, R, T, B = 112, 910, 108, 500

    def __init__(self, title: str, subtitle: str, footer: str):
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.W}" height="{self.H}" viewBox="0 0 {self.W} {self.H}" role="img" aria-label="{escape(title, quote=True)}">',
                      '<style>text{font-family:Arial,Helvetica,sans-serif;fill:currentColor}.title{font-size:25px;font-weight:700}.subtitle{font-size:14px;opacity:.72}.footnote{font-size:12px;opacity:.66}.tick{font-size:13px;opacity:.83}.axislabel{font-size:15px;font-weight:600}.legend{font-size:14px}.value{font-size:14px;font-weight:600}</style>']
        self.text(44, 44, title, 'title')
        self.text(44, 72, subtitle, 'subtitle')
        self.text(44, 575, footer, 'footnote')

    def raw(self, s: str):
        self.parts.append(s)

    def text(self, x, y, t, cls='', anchor=None, rotate=None):
        attrs = f' x="{x:.2f}" y="{y:.2f}"'
        if cls: attrs += f' class="{cls}"'
        if anchor: attrs += f' text-anchor="{anchor}"'
        if rotate is not None: attrs += f' transform="rotate({rotate} {x:.2f} {y:.2f})"'
        self.raw(f'<text{attrs}>{escape(str(t))}</text>')

    def line(self, x1, y1, x2, y2, thick=1.5, opacity=1, dash=''):
        d = f' stroke-dasharray="{dash}"' if dash else ''
        self.raw(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="currentColor" fill="none" stroke-width="{thick}" opacity="{opacity}"{d}/>')

    def rect(self, x, y, w, h, opacity=1):
        self.raw(f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" fill="currentColor" opacity="{opacity}"/>')

    def point(self, x, y, shape='circle'):
        if shape == 'circle':
            self.raw(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="white" stroke="currentColor" stroke-width="2.6"/>')
        else:
            self.raw(f'<rect x="{x-5:.2f}" y="{y-5:.2f}" width="10" height="10" fill="white" stroke="currentColor" stroke-width="2.6"/>')

    def axes(self, xlabel, ylabel, min_y, max_y, y_ticks, xlabels: list[tuple[float, str]], tick_fmt=lambda y: f'{y:g}'):
        self.min_y, self.max_y = min_y, max_y
        for value in y_ticks:
            y = self.map_y(value)
            self.line(self.L, y, self.R, y, thick=1, opacity=.12)
            self.text(self.L-16, y+5, tick_fmt(value), 'tick', 'end')
        self.line(self.L, self.T, self.L, self.B, opacity=.65)
        self.line(self.L, self.B, self.R, self.B, opacity=.65)
        for x, label in xlabels:
            self.text(x, self.B+28, label, 'tick', 'middle')
        self.text((self.L+self.R)/2, 555, xlabel, 'axislabel', 'middle')
        self.text(26, (self.T+self.B)/2, ylabel, 'axislabel', 'middle', -90)

    def map_y(self, value):
        return self.B-(value-self.min_y)/(self.max_y-self.min_y)*(self.B-self.T)

    def map_x_number(self, x, min_x, max_x):
        return self.L+(x-min_x)/(max_x-min_x)*(self.R-self.L)

    def series(self, coords, shape, dash='', offset_label=0):
        for i in range(len(coords)-1):
            self.line(*coords[i], *coords[i+1], thick=2.7, dash=dash)
        for x,y in coords:
            self.point(x,y,shape)

    def legend(self, entries):
        x, y = 618, 120
        for i,(label,shape,dash) in enumerate(entries):
            yy=y+i*29
            self.line(x, yy, x+32, yy, thick=2.5, dash=dash)
            self.point(x+16,yy,shape)
            self.text(x+43,yy+5,label,'legend')

    def save(self, path: Path):
        path.write_text('\n'.join(self.parts+['</svg>']), encoding='utf-8')
        print('Saved', path)


def baseline_throughput(data, out):
    g=SVG('Aggregate output throughput scales with concurrency',
          'Initial baseline · Qwen3-4B on NVIDIA L4 · vLLM 0.21.0',
          '5 measured batches per workload/concurrency · throughput is a mean of batch-level rates')
    xvals=[1,4,8]
    coords_x=[g.map_x_number(x,1,8) for x in xvals]
    g.axes('Concurrent requests','Output tokens / second',0,225,[0,50,100,150,200],list(zip(coords_x,map(str,xvals))))
    for name,shape,dash,dx,offset in [('short','circle','',-6,-12),('long','square','9 6',6,18)]:
        yvals=[statistics.mean(b['output_tokens_per_sec'] for b in groups(data,name,c)) for c in xvals]
        coords=[(x,g.map_y(y)) for x,y in zip(coords_x,yvals)]
        g.series(coords,shape,dash)
        for (x,y),value in zip(coords,yvals):
            g.text(x+dx,y+offset,f'{value:.1f}','value','middle')
    g.legend([('Short · 256 input tokens','circle',''),('Long · 2,048 input tokens','square','9 6')])
    g.save(out/'01_baseline_throughput.svg')


def baseline_ttft(data,out):
    g=SVG('Higher concurrency increases median first-token latency',
          'Initial baseline · client-observed TTFT · cold starts excluded',
          'p50 pooled across successful requests in 5 batches per condition · synthetic prompts')
    xvals=[1,4,8]
    xs=[g.map_x_number(x,1,8) for x in xvals]
    g.axes('Concurrent requests','TTFT p50 (ms)',0,2350,[0,500,1000,1500,2000],list(zip(xs,map(str,xvals))))
    for name,shape,dash,offset in [('short','circle','',-13),('long','square','9 6',23)]:
        vals=[pct(get_ttfts(groups(data,name,c)),50) for c in xvals]
        coords=[(x,g.map_y(y)) for x,y in zip(xs,vals)]
        g.series(coords,shape,dash)
        for (x,y),val in zip(coords,vals):
            g.text(x,y+offset,f'{val:.0f}','value','middle')
    g.legend([('Short · 256 input tokens','circle',''),('Long · 2,048 input tokens','square','9 6')])
    g.save(out/'02_baseline_ttft.svg')


def budget_effect(control,candidate,out):
    labels,changes=[],[]
    for work in ('short','long'):
        for p in (50,95):
            v0=pct(get_ttfts(groups(control,work,8)),p)
            v1=pct(get_ttfts(groups(candidate,work,8)),p)
            labels.append(f'{work.capitalize()} p{p}')
            changes.append(100*(v1-v0)/v0)
    g=SVG('Larger scheduling budget did not improve first-token latency',
          'Relative change when raising max_num_batched_tokens from 2,048 to 4,096',
          '20 measured batches × 8 requests per workload/config · positive = worse · fixed run order: descriptive only')
    xs=[g.L+100+i*190 for i in range(4)]
    g.axes('Workload / TTFT percentile','Change vs. 2,048 (%)',0,16,[0,4,8,12,16],list(zip(xs,labels)),lambda y:f'+{y:g}%' if y else '0%')
    for x,val in zip(xs,changes):
        y=g.map_y(val)
        g.rect(x-35,y,70,g.B-y,opacity=.85)
        g.text(x,y-13,f'+{val:.2f}%' if val>=0 else f'{val:.2f}%','value','middle')
    g.save(out/'03_budget_ttft_delta.svg')


def ranked_ttft(control,candidate,out):
    g=SVG('Long-prompt first-token arrivals are staggered',
          'Concurrency 8 · 2,048-token inputs · 20 batches per scheduler budget',
          'Markers = median at each sorted rank · whiskers = interquartile range · sorted ranks do not preserve request identity')
    xs=[g.map_x_number(i,1,8) for i in range(1,9)]
    g.axes('First-token arrival rank (fastest to slowest)','TTFT (ms)',0,3400,[0,500,1000,1500,2000,2500,3000],list(zip(xs,map(str,range(1,9)))))
    for data,shape,dash,xoffset in ((control,'circle','',-6),(candidate,'square','9 6',6)):
        per_trial=[]
        for batch in groups(data,'long',8):
            vals=sorted(r['ttft_ms'] for r in batch['results'] if r['success'] and r['ttft_ms'] is not None)
            if len(vals)!=8: raise ValueError('Expected 8 valid TTFTs per batch')
            per_trial.append(vals)
        med=[pct([t[i] for t in per_trial],50) for i in range(8)]
        lows=[pct([t[i] for t in per_trial],25) for i in range(8)]
        highs=[pct([t[i] for t in per_trial],75) for i in range(8)]
        points=[(x+xoffset,g.map_y(v)) for x,v in zip(xs,med)]
        g.series(points,shape,dash)
        for x,low,hi in zip(xs,lows,highs):
            xx=x+xoffset
            y1,y2=g.map_y(low),g.map_y(hi)
            g.line(xx,y1,xx,y2,thick=1.6)
            g.line(xx-4,y1,xx+4,y1,thick=1.6)
            g.line(xx-4,y2,xx+4,y2,thick=1.6)
    g.legend([('2,048-token budget','circle',''),('4,096-token budget','square','9 6')])
    g.save(out/'04_long_ttft_ranked.svg')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir',type=Path,default=Path(__file__).resolve().parent.parent/'results')
    parser.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent)
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    data={name:read_batches(args.results_dir/fname) for name,fname in FILES.items()}
    baseline_throughput(data['baseline'],args.output_dir)
    baseline_ttft(data['baseline'],args.output_dir)
    budget_effect(data['control'],data['candidate'],args.output_dir)
    ranked_ttft(data['control'],data['candidate'],args.output_dir)

if __name__=='__main__':
    main()
