#!/usr/bin/env python
"""Diagnostic: near-surface air temperature against ERA5, and soil beneath it.

This is not a manuscript figure. It exists to answer whether a case is tracking
the reanalysis it was built from, which the case-study figure cannot show
because it plots no reference.

Top row is the model's lowest-level air temperature against ERA5's 2 m
temperature at the same site. Bottom row is the column-mean soil temperature
underneath. Both are daily means, because a 15 K bias is what is being looked
for and the diurnal cycle only obscures it at this length.

    python paper/make_diagnostic_figure.py [--scm-root PATH] [--era5-dir PATH]

``--era5-dir`` holds the raw ``*_sfc.nc`` extractions the cases were built
from, which is where the ERA5 reference comes from.
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

SITES = [
    ("US-Whs", "US-Whs  desert shrubland, Jan 2019", "#2a78d6"),
    ("US-MMS", "US-MMS  temperate forest, Jun-Jul 2019", "#eb6834"),
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

SOIL_THICKNESS = np.array([0.10, 0.30, 0.60, 1.00])
SPINUP_STEPS = 2


def daily_mean(series):
    """Collapse an hourly series to whole-day means."""
    ndays = len(series) // 24
    return np.array([series[i * 24:(i + 1) * 24].mean() for i in range(ndays)])


def column_mean_soil_temperature(ds):
    soil = np.asarray(ds["soil_T"].squeeze().values)
    return (soil * SOIL_THICKNESS).sum(axis=1) / SOIL_THICKNESS.sum()


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
    parser.add_argument("--era5-dir", default=None,
                        help="Directory holding the raw {case}_sfc.nc "
                             "extractions, searched recursively.")
    args = parser.parse_args()

    run_dir = os.path.join(args.scm_root, "scm", "run")

    def output(prefix, site):
        path = os.path.join(run_dir, f"output_{prefix}_{site}_SCM_GFS_v16",
                            "output.nc")
        return xr.open_dataset(path) if os.path.exists(path) else None

    def era5_reference(site):
        """ERA5 2 m temperature at the stencil centre, if the raw file is here."""
        if args.era5_dir is None:
            return None
        for root, _, files in os.walk(args.era5_dir):
            for name in files:
                if name.endswith("_sfc.nc") and site in name:
                    with xr.open_dataset(os.path.join(root, name)) as ds:
                        if "t2m" not in ds:
                            continue
                        return np.asarray(
                            ds["t2m"].isel(latitude=1, longitude=1).values)
        return None

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.0))

    for col, (site, label, color) in enumerate(SITES):
        era5_run = output("era5land", site)
        template_run = output("fluxnet", site)
        if era5_run is None:
            continue

        air = daily_mean(np.asarray(era5_run["T"].squeeze().values)[:, 0])
        soil = daily_mean(column_mean_soil_temperature(era5_run))
        days = np.arange(1, len(air) + 1)

        ax = axes[0, col]
        ax.plot(days, air, color=color, linewidth=1.8,
                label="ERA5 initial state", zorder=4)
        if template_run is not None:
            tpl = daily_mean(
                np.asarray(template_run["T"].squeeze().values)[:, 0])
            ax.plot(days[:len(tpl)], tpl, color=color, linewidth=1.0,
                    linestyle=(0, (3, 2.5)), label="GABLS3 template", zorder=3)

        reference = era5_reference(site)
        if reference is not None:
            ref = daily_mean(reference[:len(air) * 24])
            ax.plot(days[:len(ref)], ref, color=INK, linewidth=1.2,
                    label="ERA5 2 m temperature", zorder=5)

        style(ax, f"({'ab'[col]})  {site} air temperature", "day of run",
              "K" if col == 0 else None)
        ax.legend(frameon=False, fontsize=6.5, loc="best", handlelength=1.8,
                  borderpad=0.2)

        ax = axes[1, col]
        ax.plot(days, soil, color=color, linewidth=1.8, zorder=4)
        if template_run is not None:
            tpl_soil = daily_mean(column_mean_soil_temperature(template_run))
            ax.plot(days[:len(tpl_soil)], tpl_soil, color=color, linewidth=1.0,
                    linestyle=(0, (3, 2.5)), zorder=3)
        style(ax, f"({'cd'[col]})  {site} column-mean soil temperature",
              "day of run", "K" if col == 0 else None)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.45, wspace=0.28)

    out = os.path.join(HERE, "diagnostic_temperature.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"wrote {out}")

    # The number that matters: is the case tracking the reanalysis at all?
    for site, _, _ in SITES:
        era5_run = output("era5land", site)
        reference = era5_reference(site)
        if era5_run is None or reference is None:
            continue
        air = daily_mean(np.asarray(era5_run["T"].squeeze().values)[:, 0])
        ref = daily_mean(reference[:len(air) * 24])
        n = min(len(air), len(ref))
        bias = air[:n].mean() - ref[:n].mean()
        rmse = float(np.sqrt(((air[:n] - ref[:n]) ** 2).mean()))
        corr = float(np.corrcoef(air[:n], ref[:n])[0, 1])
        print(f"{site}: air-temperature bias {bias:+.1f} K, "
              f"daily RMSE {rmse:.1f} K, correlation {corr:.2f}")


if __name__ == "__main__":
    main()
