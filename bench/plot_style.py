BLUE = "#0072B2"
ORANGE = "#C55A11"
INK = "#17212b"

FIGURE_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": 10,
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.labelsize": 11,
    "axes.labelweight": "semibold",
    "axes.titlesize": 11,
    "axes.titleweight": "semibold",
    "axes.titlelocation": "left",
    "axes.edgecolor": INK,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "xtick.color": INK,
    "ytick.color": INK,
    "grid.color": INK,
    "grid.alpha": 0.12,
    "grid.linewidth": 0.7,
    "grid.linestyle": "-",
    "lines.linestyle": "-",
    "lines.linewidth": 1.5,
    "legend.frameon": False,
    "figure.titlesize": 18,
    "figure.titleweight": "bold",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "svg.fonttype": "none",
}

def label_figure(figure, title: str, subtitle: str, footnote: str):
    figure.suptitle(title, x=0.045, y=0.975, ha="left")
    figure.text(0.045, 0.92, subtitle, fontsize=10, alpha=0.72)
    figure.text(0.045, 0.025, footnote, fontsize=9, alpha=0.66)
