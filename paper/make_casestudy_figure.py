#!/usr/bin/env python
"""Draw the case-study figure (Figure 2) for the JOSS manuscript.

Top row: what the tool writes into the DEPHY driver.
Bottom row: what CCPP-SCM produces from it.

    python paper/make_casestudy_figure.py [--scm-root PATH]
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SCM_ROOT = "/glade/u/home/andrbenn/workspace/wpo/scm/ccpp-scm"

# Categorical slots 1 and 2 of the reference palette, assigned in fixed order.
SITES = [
    ("US-Whs", "US-Whs  desert shrubland, 1 Jan 2019", "#2a78d6"),
    ("US-MMS", "US-MMS  temperate forest, 15 Jun 2019", "#eb6834"),
]

INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#dedddb"
SURFACE = "#ffffff"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.edgecolor": INK_MUTED,
    "axes.labelcolor": INK,
    "axes.titlesize": 8.5,
    "axes.linewidth": 0.8,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})

LINEWIDTH = 1.8
#: Profiles are cut at 200 hPa; above that theta_l runs to ~1900 K and would
#: flatten everything of interest in the troposphere.
P_TOP = 20000.0


def style(ax, title, xlabel, ylabel=None):
    ax.set_title(title, color=INK, fontweight="bold", pad=6)
    ax.set_xlabel(xlabel, color=INK_MUTED)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_MUTED)
    ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scm-root", default=DEFAULT_SCM_ROOT)
    args = parser.parse_args()

    case_dir = os.path.join(args.scm_root, "scm", "data", "processed_case_input")
    run_dir = os.path.join(args.scm_root, "scm", "run")

    drivers, outputs = {}, {}
    for site, _, _ in SITES:
        drivers[site] = xr.open_dataset(
            os.path.join(case_dir, f"fluxnet_{site}_SCM_driver.nc"),
            decode_times=False)
        outputs[site] = xr.open_dataset(
            os.path.join(run_dir, f"output_fluxnet_{site}_SCM_GFS_v16",
                         "output.nc"))

    fig, axes = plt.subplots(2, 3, figsize=(7.4, 5.0))

    # ------------------------------------------------------------------
    # Top row — DEPHY driver contents (the tool's output)
    # ------------------------------------------------------------------
    for site, label, color in SITES:
        ds = drivers[site]
        p_hpa = ds["lev"].values / 100.0
        keep = ds["lev"].values >= P_TOP

        axes[0, 0].plot(ds["thetal"].values[0][keep], p_hpa[keep],
                        color=color, linewidth=LINEWIDTH, label=label, zorder=3)
        axes[0, 1].plot(ds["qt"].values[0][keep] * 1000.0, p_hpa[keep],
                        color=color, linewidth=LINEWIDTH, label=label, zorder=3)

    style(axes[0, 0], "(a)  Initial $\\theta_l$", "K", "pressure (hPa)")
    style(axes[0, 1], "(b)  Initial $q_t$", "g kg$^{-1}$")
    for ax in (axes[0, 0], axes[0, 1]):
        ax.invert_yaxis()
        ax.set_ylim(1000, 200)

    # Soil temperature: identical between sites, because it comes from the
    # template rather than the site. Offset the second line so the overlap is
    # visible rather than hidden.
    ax = axes[0, 2]
    for index, (site, label, color) in enumerate(SITES):
        ds = drivers[site]
        depth = ds["soil_depth"].values
        stc = ds["stc"].values[0]
        ax.plot(stc, depth, color=color, linewidth=LINEWIDTH,
                linestyle="-" if index == 0 else (0, (3, 2.5)),
                marker="o", markersize=4, label=label, zorder=3 + index)
    style(ax, "(c)  Initial soil $T$", "K", "depth (m)")
    ax.invert_yaxis()
    ax.text(0.5, 0.06,
            "identical — inherited\nfrom the GABLS3 template",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=7, color=INK_MUTED, style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=SURFACE,
                      edgecolor=GRID, linewidth=0.6), zorder=5)

    # ------------------------------------------------------------------
    # Bottom row — CCPP-SCM output
    # ------------------------------------------------------------------
    panels = [
        (axes[1, 0], "shf", "(d)  Sensible heat flux", "W m$^{-2}$"),
        (axes[1, 1], "lhf", "(e)  Latent heat flux", "W m$^{-2}$"),
        (axes[1, 2], "hpbl", "(f)  Boundary-layer height", "m"),
    ]
    for ax, varname, title, units in panels:
        for site, label, color in SITES:
            ds = outputs[site]
            hours = ds["time_inst"].values / 3600.0
            ax.plot(hours, np.asarray(ds[varname].squeeze().values),
                    color=color, linewidth=LINEWIDTH, label=label, zorder=3)
        style(ax, title, "hours since 00 UTC",
              units if varname != "hpbl" else "m")
        ax.set_xlim(0, 24)
        ax.set_xticks([0, 6, 12, 18, 24])

    axes[1, 0].axhline(0.0, color=INK_MUTED, linewidth=0.7, zorder=2)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               fontsize=8, labelcolor=INK, bbox_to_anchor=(0.5, -0.005))

    fig.tight_layout(rect=(0, 0.055, 1, 1))
    fig.subplots_adjust(hspace=0.42, wspace=0.38)

    out = os.path.join(HERE, "case_study.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"wrote {out}")

    # Numbers quoted in the manuscript text.
    for site, label, _ in SITES:
        ds = outputs[site]
        shf = np.asarray(ds["shf"].squeeze().values)[1:]
        lhf = np.asarray(ds["lhf"].squeeze().values)[1:]
        ef = lhf.mean() / (lhf.mean() + shf.mean())
        print(f"{site}: mean SHF {shf.mean():6.1f}  mean LHF {lhf.mean():6.1f}"
              f"  peak LHF {lhf.max():6.1f}  evaporative fraction {ef:.2f}"
              f"  peak PBL {float(ds['hpbl'].max()):.0f} m")


if __name__ == "__main__":
    main()
