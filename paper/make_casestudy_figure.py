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

#: Output steps dropped from the flux panels and the quoted statistics. Step 0
#: is the initial state, and step 1 carries a transient left by starting Noah
#: from a surface that HTESSEL equilibrated.
SPINUP_STEPS = 2

#: Downward shortwave at the surface, W m-2, above which an hour counts as
#: daylight for the evaporative fraction.
DAYLIGHT_SW = 20.0

#: Thickness of each Noah soil layer, m, used to depth-weight the column mean.
SOIL_THICKNESS = np.array([0.10, 0.30, 0.60, 1.00])

#: Case prefix carrying the template-soil initialisation, drawn alongside the
#: ERA5-initialised run in the soil panel so the effect of the initial state is
#: visible rather than asserted.
COMPARISON_PREFIX = "fluxnet"

#: Shared x-axis label for every time series panel.
TIME_LABEL = "hours since 00 UTC on the start date"


def column_mean_soil_temperature(ds):
    """Depth-weighted mean soil temperature over the four Noah layers."""
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
    parser.add_argument("--case-prefix", default="era5land",
                        help="Case-name prefix to plot. 'era5land' are the "
                             "cases with the land surface initialised from "
                             "ERA5; 'fluxnet' are the earlier template-soil "
                             "cases.")
    args = parser.parse_args()

    case_dir = os.path.join(args.scm_root, "scm", "data", "processed_case_input")
    run_dir = os.path.join(args.scm_root, "scm", "run")

    drivers, outputs = {}, {}
    for site, _, _ in SITES:
        drivers[site] = xr.open_dataset(
            os.path.join(case_dir,
                         f"{args.case_prefix}_{site}_SCM_driver.nc"),
            decode_times=False)
        outputs[site] = xr.open_dataset(
            os.path.join(run_dir,
                         f"output_{args.case_prefix}_{site}_SCM_GFS_v16",
                         "output.nc"))

    # The soil panel draws the template-soil run alongside, where it exists.
    comparison = {}
    if args.case_prefix != COMPARISON_PREFIX:
        for site, _, _ in SITES:
            path = os.path.join(
                run_dir, f"output_{COMPARISON_PREFIX}_{site}_SCM_GFS_v16",
                "output.nc")
            if os.path.exists(path):
                comparison[site] = xr.open_dataset(path)

    # Time axes follow the length of the run, so the same script serves a
    # two-day case and a month-long one.
    duration = max(float(ds["time_inst"].values[-1]) / 3600.0
                   for ds in outputs.values())
    tick_step = 24.0 if duration <= 144 else 168.0 if duration <= 840 else 240.0
    xticks = np.arange(0.0, duration + tick_step * 0.5, tick_step)
    # Hourly fluxes over a month are a dense diurnal band, so the stroke has to
    # thin out or the envelope fills solid.
    series_lw = LINEWIDTH if duration <= 144 else 0.6

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

    # Soil temperature through the run. The initial state is what this package
    # newly supplies, so the panel shows how long it goes on mattering rather
    # than only what it was at hour zero. The template-soil runs are still
    # loaded, because the statistics printed below quote them, but they are not
    # drawn: the panel is about the two sites differing from each other.
    ax = axes[0, 2]
    for site, label, color in SITES:
        hours = outputs[site]["time_inst"].values / 3600.0
        ax.plot(hours, column_mean_soil_temperature(outputs[site]),
                color=color, linewidth=LINEWIDTH, label=label, zorder=4)

    style(ax, "(c)  Column-mean soil $T$", TIME_LABEL, "K")
    ax.set_xlim(0, duration)
    ax.set_xticks(xticks)

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
            hours = ds["time_inst"].values[SPINUP_STEPS:] / 3600.0
            series = np.asarray(ds[varname].squeeze().values)[SPINUP_STEPS:]
            ax.plot(hours, series, color=color, linewidth=series_lw,
                    label=label, zorder=3)
        style(ax, title, TIME_LABEL, units if varname != "hpbl" else "m")
        ax.set_xlim(0, duration)
        ax.set_xticks(xticks)

    axes[1, 0].axhline(0.0, color=INK_MUTED, linewidth=0.7, zorder=2)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               fontsize=8, labelcolor=INK, bbox_to_anchor=(0.5, -0.005))

    fig.tight_layout(rect=(0, 0.055, 1, 1))
    fig.subplots_adjust(hspace=0.42, wspace=0.38)

    # The manuscript figure is the ERA5-initialised one; plotting the earlier
    # template-soil cases for comparison must not overwrite it.
    stem = "case_study" if args.case_prefix == "era5land" \
        else f"case_study_{args.case_prefix}"
    out = os.path.join(HERE, f"{stem}.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"wrote {out}")

    # Numbers quoted in the manuscript text, taken over the whole run.
    #
    # Statistics are restricted to daylight hours. The nocturnal sensible heat
    # flux is strongly negative, so an all-hours ratio has a denominator that
    # approaches zero and stops being a fraction at all. Daylight is defined by
    # downward shortwave at the surface rather than by clock time, so it adapts
    # to site and season.
    print(f"--- whole run, daylight hours ({args.case_prefix}) ---")
    for site, label, _ in SITES:
        ds = outputs[site]
        shf = np.asarray(ds["shf"].squeeze().values)[SPINUP_STEPS:]
        lhf = np.asarray(ds["lhf"].squeeze().values)[SPINUP_STEPS:]
        pbl = np.asarray(ds["hpbl"].squeeze().values)[SPINUP_STEPS:]
        swd = np.asarray(ds["sfc_dwn_sw"].squeeze().values)[SPINUP_STEPS:]

        day = swd > DAYLIGHT_SW
        ef = lhf[day].sum() / (lhf[day] + shf[day]).sum()
        print(f"{site}: daylight mean SHF {shf[day].mean():7.1f} "
              f" mean LHF {lhf[day].mean():7.1f}"
              f"  peak LHF {lhf.max():7.1f}"
              f"  evaporative fraction {ef:5.2f}"
              f"  peak PBL {pbl.max():.0f} m")

    print(f"--- soil and drift ({args.case_prefix}) ---")
    for site, label, _ in SITES:
        ds = outputs[site]
        air = np.asarray(ds["T"].squeeze().values)[:, 0]
        soil = column_mean_soil_temperature(ds)
        ndays = len(air) // 24
        first = air[SPINUP_STEPS:24].mean()
        last = air[(ndays - 1) * 24:ndays * 24].mean()
        print(f"{site}: column-mean soil T {soil[SPINUP_STEPS]:.1f} -> "
              f"{soil[-1]:.1f} K over {ndays} days; "
              f"lowest-level air T drift {last - first:+.1f} K")
        if site in comparison:
            tpl = column_mean_soil_temperature(comparison[site])
            n = min(len(soil), len(tpl))
            print(f"{'':8s} template-initialised soil differs by "
                  f"{soil[SPINUP_STEPS] - tpl[SPINUP_STEPS]:+.2f} K at the "
                  f"start and {soil[n - 1] - tpl[n - 1]:+.2f} K at the end")

if __name__ == "__main__":
    main()
