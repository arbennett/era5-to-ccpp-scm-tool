#!/usr/bin/env python
"""Draw the workflow diagram (Figure 1) for the JOSS manuscript.

    python paper/make_workflow_figure.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))

# Ink and surface tokens; series slots 1 and 2 from the reference palette.
INK = "#0b0b0b"
INK_MUTED = "#52514e"
SURFACE = "#ffffff"
STAGE_EDGE = "#2a78d6"
STAGE_FILL = "#eaf2fc"
SOURCE_EDGE = "#52514e"
SOURCE_FILL = "#f2f1ef"
MODEL_EDGE = "#eb6834"
MODEL_FILL = "#fdeee7"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "savefig.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
})


def box(ax, x, y, w, h, title, body, edge, fill,
        title_size=9.0, outputs=None):
    """A stage box: bold title, muted body, and an italic outputs line
    pinned to the bottom edge so it never collides with a neighbour."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.4, edgecolor=edge, facecolor=fill, zorder=2,
    ))
    ax.text(x + w / 2, y + h - 0.045, title, ha="center", va="top",
            fontsize=title_size, fontweight="bold", color=INK,
            linespacing=1.35, zorder=3)
    if body:
        body_top = y + h - 0.115 - 0.052 * title.count("\n")
        ax.text(x + w / 2, body_top, body, ha="center", va="top",
                fontsize=7.2, color=INK_MUTED, linespacing=1.5, zorder=3)
    if outputs:
        ax.text(x + w / 2, y + 0.030, outputs, ha="center", va="bottom",
                fontsize=6.8, color=edge, style="italic",
                linespacing=1.4, zorder=3)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle="-|>", mutation_scale=11,
        linewidth=1.3, color=INK_MUTED,
        shrinkA=0, shrinkB=0, zorder=1,
    ))


def main():
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    ax.set_xlim(-0.01, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    row_y, row_h = 0.40, 0.44
    w = 0.203
    xs = [0.010, 0.259, 0.508, 0.757]

    # --- Data source ------------------------------------------------------
    box(ax, xs[0], row_y, w, row_h,
        "ERA5 archive",
        "s3://nsf-ncar-era5\nor the GLADE mount\n\n"
        "public and anonymous:\nno account, no API key,\nno request queue",
        SOURCE_EDGE, SOURCE_FILL, title_size=8.6)

    # --- Pipeline stages --------------------------------------------------
    box(ax, xs[1], row_y, w, row_h,
        "download_era5",
        "3×3 stencil at the site\n\npressure levels, surface,\n"
        "accumulated fluxes\n\nparallel across variables",
        STAGE_EDGE, STAGE_FILL,
        outputs="→ *_pl.nc,  *_sfc.nc")

    box(ax, xs[2], row_y, w, row_h,
        "convert_forcings",
        "geostrophic winds\nadvective tendencies\n"
        "ω → w conversion\nθ$_l$, q$_t$ from T and q\nradiative heating",
        STAGE_EDGE, STAGE_FILL,
        outputs="→ forcings.nc")

    box(ax, xs[3], row_y, w, row_h,
        "convert_to_dephy",
        "DEPHY driver file\nand case-config namelist\n\n"
        "static land state from\nthe GABLS3 template",
        STAGE_EDGE, STAGE_FILL,
        outputs="→ *_SCM_driver.nc,  *.nml")

    mid = row_y + row_h / 2
    for left in range(3):
        arrow(ax, xs[left] + w, mid, xs[left + 1] - 0.008, mid)

    # --- Target model -----------------------------------------------------
    model_y, model_h = 0.055, 0.20
    model_x, model_w = 0.508, 0.452
    box(ax, model_x, model_y, model_w, model_h,
        "CCPP-SCM",
        "./run_scm.py -c <case> -s SCM_GFS_v16",
        MODEL_EDGE, MODEL_FILL)

    down_x = xs[3] + w / 2
    arrow(ax, down_x, row_y - 0.004, down_x, model_y + model_h + 0.004)

    # --- The one-command path --------------------------------------------
    band_y = 0.895
    ax.add_patch(FancyBboxPatch(
        (xs[1], band_y), xs[3] + w - xs[1], 0.070,
        boxstyle="round,pad=0.008,rounding_size=0.02",
        linewidth=1.2, edgecolor=STAGE_EDGE, facecolor=SURFACE,
        linestyle=(0, (4, 2.5)), zorder=2,
    ))
    ax.text((xs[1] + xs[3] + w) / 2, band_y + 0.035,
            "run_full_pipeline  —  all three stages in one command",
            ha="center", va="center", fontsize=8.0,
            color=STAGE_EDGE, fontweight="bold", zorder=3)

    # add a bit of padding around the overall figure
    fig.tight_layout(pad=0.3, )
    out = os.path.join(HERE, "workflow.png")
    fig.savefig(out, dpi=300, bbox_inches="tight", )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
