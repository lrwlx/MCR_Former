## -*- coding: utf-8 -*-
"""Redraw Figure 1 as a production-style vector schematic (PDF/SVG + 600 dpi PNG)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

BLUE, LIGHT, GREY, RED, GREEN = "#1f4e79", "#9dc3e6", "#595959", "#c00000", "#2e7d32"

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "Fig1_redrawn")


def box(ax, x, y, w, h, text, fc="white", ec=GREY, fs=6.2, weight="normal", lw=0.8, pad=0.02):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=%.3f" % pad,
                                linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color="black", zorder=3, fontweight=weight, linespacing=1.35)


def arrow(ax, p0, p1, color=GREY, lw=0.9, style="-|>", rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=8,
                                 linewidth=lw, color=color, zorder=4,
                                 connectionstyle="arc3,rad=%.2f" % rad))


def sphere(ax, cx, cy, r):
    ax.add_patch(Circle((cx, cy), r, fill=False, lw=0.8, ec=GREY, zorder=2))
    ax.add_patch(Ellipse((cx, cy), 2 * r, 2 * r * 0.34, fill=False, lw=0.6, ec=LIGHT, zorder=2))
    ax.add_patch(Ellipse((cx, cy), 2 * r * 0.34, 2 * r, fill=False, lw=0.6, ec=LIGHT, zorder=2))
    ax.plot([cx], [cy + r], marker="o", ms=2.6, color=RED, zorder=3)


def main():
    fig = plt.figure(figsize=(7.2, 6.2))
    gs = fig.add_gridspec(3, 1, height_ratios=[0.95, 1.5, 1.0], hspace=0.28,
                          left=0.02, right=0.98, top=0.985, bottom=0.02)

    # ---------------- panel a: representation ----------------
    ax = fig.add_subplot(gs[0])
    ax.set_xlim(0, 10); ax.set_ylim(0, 2.6); ax.axis("off")
    ax.text(0.0, 2.42, "a", fontsize=8, fontweight="bold", va="top")
    ax.text(0.22, 2.42, "Minimal complete representation", fontsize=7.4, va="top")

    ax.plot([0.7, 0.7], [0.35, 1.5], color=GREY, lw=1.0)
    ax.plot([0.7], [0.35], marker="o", ms=3, color=BLUE)
    ax.plot([0.7], [1.5], marker="o", ms=3, color=BLUE)
    ax.text(0.55, 1.72, "180°", fontsize=6, ha="right")
    ax.text(0.55, 0.28, "0°", fontsize=6, ha="right")
    ax.text(0.95, 0.9, "raw angle\n$\\theta\\in[0,180^\\circ)$", fontsize=6.2, va="center")
    ax.text(0.7, 0.05, "the seam is artificial", fontsize=5.6, ha="center", color=RED)

    arrow(ax, (2.35, 0.92), (3.05, 0.92))
    ax.text(2.7, 1.02, "MCR", fontsize=6, ha="center")

    ax.add_patch(Circle((3.85, 0.92), 0.62, fill=False, lw=1.0, ec=BLUE))
    for a, c in [(40, RED), (220, RED)]:
        ax.plot([3.85 + 0.62 * np.cos(np.radians(a))], [0.92 + 0.62 * np.sin(np.radians(a))],
                marker="o", ms=3, color=c, zorder=3)
    ax.text(3.85, 1.72, "unit circle", fontsize=6.2, ha="center")
    ax.text(3.85, -0.06, "$(\\sin 2\\theta,\\ \\cos 2\\theta)$", fontsize=6.2, ha="center")
    ax.text(5.0, 1.2, "0° and 180°\ncoincide", fontsize=5.8, va="center", color=RED)

    ax.add_patch(Rectangle((6.5, 0.5), 0.85, 0.85, fill=False, lw=0.9, ec=GREY))
    ax.text(6.92, 1.5, "spherical angles", fontsize=6.2, ha="center")
    ax.text(6.92, 0.2, "$(\\theta,\\phi)$", fontsize=6.2, ha="center")
    arrow(ax, (7.55, 0.92), (8.15, 0.92))
    sphere(ax, 8.75, 0.92, 0.62)
    ax.text(8.75, 1.72, "unit sphere", fontsize=6.2, ha="center")
    ax.text(8.75, -0.06, "$(\\sin\\theta\\cos\\phi,\\ \\sin\\theta\\sin\\phi,\\ \\cos\\theta)$", fontsize=5.6, ha="center")

    # ---------------- panel b: architecture ----------------
    ax = fig.add_subplot(gs[1])
    ax.set_xlim(0, 10); ax.set_ylim(0, 4.2); ax.axis("off")
    ax.text(0.0, 4.05, "b", fontsize=8, fontweight="bold", va="top")
    ax.text(0.22, 4.05, "Dual-path transformer with shared weights", fontsize=7.4, va="top")

    for i in range(4):
        box(ax, 0.15, 2.75 - i * 0.42, 0.85, 0.3, "$\\theta_%d$" % (i + 1), fs=6)
    ax.text(0.57, 1.05, "$\\vdots$", fontsize=7, ha="center")
    box(ax, 0.15, 0.55, 0.85, 0.3, "pad", fs=5.6, fc="#f2f2f2")
    ax.text(0.57, 0.32, "ply angles", fontsize=5.8, ha="center")
    arrow(ax, (1.05, 1.8), (1.42, 1.8))
    box(ax, 1.45, 1.35, 1.15, 0.9, "MCR\nencoding\n$(\\sin 2\\theta,\\cos 2\\theta)$", fs=5.8, fc="#eaf1f8")

    arrow(ax, (2.65, 1.8), (3.0, 1.8))
    box(ax, 3.02, 1.35, 0.95, 0.9, "positional\nencoding", fs=5.8)

    arrow(ax, (4.0, 1.8), (4.35, 2.55)); arrow(ax, (4.0, 1.8), (4.35, 1.05))
    ax.text(4.18, 2.75, "$X$", fontsize=6, ha="center")
    ax.text(4.18, 0.72, "$R(X)$", fontsize=6, ha="center")

    box(ax, 4.42, 2.15, 1.5, 1.0, "transformer\nencoder\n(shared weights)", fs=5.9, fc="#dbe8f5")
    box(ax, 4.42, 0.45, 1.5, 1.0, "same encoder\non the reversed\nsequence", fs=5.9, fc="#dbe8f5")
    arrow(ax, (5.95, 2.65), (6.35, 2.65)); arrow(ax, (5.95, 0.95), (6.35, 0.95))

    box(ax, 6.4, 1.45, 1.35, 0.85, "symmetric\nfusion\n$\\frac{1}{2}(E(X)+E(R(X)))$", fs=5.9, fc="#e8f3ea")
    arrow(ax, (7.78, 1.88), (8.15, 1.88))
    box(ax, 8.18, 1.45, 1.05, 0.85, "regression\nhead (MLP)", fs=5.9)
    arrow(ax, (9.25, 1.88), (9.6, 1.88))
    ax.text(9.72, 1.88, "$\\nu_{13}$", fontsize=7.4, va="center")

    # inverse-design loop, routed along the top so that no text is crossed
    box(ax, 3.0, 3.35, 4.6, 0.52,
        "inverse design: differential evolution on the continuous embedding", fs=5.8, fc="#fdeeee")
    arrow(ax, (9.75, 1.95), (9.75, 3.61), color=RED)
    arrow(ax, (9.75, 3.61), (7.62, 3.61), color=RED)
    arrow(ax, (2.98, 3.61), (2.02, 3.61), color=RED)
    arrow(ax, (2.02, 3.61), (2.02, 2.28), color=RED)
    ax.text(2.14, 2.95, "optimise the angles", fontsize=5.6, color=RED, va="center")
    ax.text(0.57, 0.05, "exact reversal invariance: averaging two shared-encoder outputs",
            fontsize=5.6, color=GREEN)

    # ---------------- panel c: application ----------------
    ax = fig.add_subplot(gs[2])
    ax.set_xlim(0, 10); ax.set_ylim(0, 2.6); ax.axis("off")
    ax.text(0.0, 2.5, "c", fontsize=8, fontweight="bold", va="top")
    ax.text(0.22, 2.5, "One model across laminate thicknesses", fontsize=7.4, va="top")

    for k in range(6):
        y = 0.55 + k * 0.22
        ax.add_patch(Polygon([[0.35, y], [1.75, y + 0.06], [1.85, y + 0.16], [0.45, y + 0.10]],
                             closed=True, facecolor="#c9d9ea" if k % 2 == 0 else "#e8eef6",
                             edgecolor=GREY, lw=0.5))
    ax.text(1.1, 0.25, "1 to 20 plies", fontsize=6, ha="center")
    arrow(ax, (1.95, 1.2), (2.35, 1.2))

    box(ax, 2.4, 1.75, 7.2, 0.55, "single model, variable-length input: no per-thickness retraining", fs=6.4, fc="#eaf1f8")
    box(ax, 2.4, 1.05, 7.2, 0.55, "cross-thickness and cross-property transfer; 20 labelled samples suffice", fs=6.4, fc="#f4f7fb")
    box(ax, 2.4, 0.35, 7.2, 0.55, "inverse design returns an auxetic stacking sequence within 1% of the optimum", fs=6.4, fc="#fdeeee")

    fig.savefig(OUT + ".png", dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT + ".pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT + ".svg", bbox_inches="tight", pad_inches=0.02)
    print("saved", OUT + ".png/.pdf/.svg")


if __name__ == "__main__":
    main()
