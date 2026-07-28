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

    # Soil temperature, taken from ERA5 at each site. The gradient reverses
    # between the two cases, which is the seasonal signal that a single
    # template cannot carry; the template profile is drawn for reference.
    ax = axes[0, 2]
    for site, label, color in SITES:
        ds = drivers[site]
        ax.plot(ds["stc"].values[0], ds["soil_depth"].values, color=color,
                linewidth=LINEWIDTH, marker="o", markersize=4, label=label,
                zorder=4)

    template = os.path.join(case_dir, "fluxnet_US-Whs_SCM_driver.nc")
    if os.path.exists(template):
        with xr.open_dataset(template, decode_times=False) as tpl:
            ax.plot(tpl["stc"].values[0], tpl["soil_depth"].values,
                    color=INK_MUTED, linewidth=1.0, linestyle=(0, (3, 2.5)),
                    marker="s", markersize=3, label="GABLS3 template",
                    zorder=3)

    style(ax, "(c)  Initial soil $T$", "K", "depth (m)")
    ax.invert_yaxis()
    handles, labels = ax.get_legend_handles_labels()
    keep = [(h, l) for h, l in zip(handles, labels) if "template" in l]
    if keep:
        ax.legend([h for h, _ in keep], [l for _, l in keep], frameon=False,
                  fontsize=6.5, loc="lower left", handlelength=1.8,
                  borderpad=0.2)

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
            ax.plot(hours, series, color=color, linewidth=LINEWIDTH,
                    label=label, zorder=3)
        style(ax, title, "hours since 00 UTC on the start date",
              units if varname != "hpbl" else "m")
        ax.set_xlim(0, 120)
        ax.set_xticks([0, 24, 48, 72, 96, 120])

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

    # Numbers quoted in the manuscript text.
    #
    # The flux comparison is taken over the first simulated day. Beyond that
    # the column drifts: these cases carry no nudging, so nothing relaxes the
    # free troposphere back toward the reanalysis, and by day three the US-MMS
    # sensible heat flux is negative even at midday. The drift diagnostic below
    # reports that directly, as the change in daily-mean air temperature at the
    # lowest model level between the first and last simulated day.
    #
    # The evaporative fraction is restricted to daylight hours. The nocturnal
    # sensible heat flux is strongly negative, so an all-hours ratio has a
    # denominator that approaches zero and stops being a fraction at all.
    print(f"--- day 1, cold-start steps excluded ({args.case_prefix}) ---")
    for site, label, _ in SITES:
        ds = outputs[site]
        shf = np.asarray(ds["shf"].squeeze().values)
        lhf = np.asarray(ds["lhf"].squeeze().values)
        pbl = np.asarray(ds["hpbl"].squeeze().values)
        swd = np.asarray(ds["sfc_dwn_sw"].squeeze().values)

        first = slice(SPINUP_STEPS, 25)
        day = swd[first] > DAYLIGHT_SW
        ef = lhf[first][day].sum() / (lhf[first] + shf[first])[day].sum()
        print(f"{site}: mean SHF {shf[first].mean():6.1f}"
              f"  mean LHF {lhf[first].mean():6.1f}"
              f"  peak LHF {lhf[first].max():6.1f}"
              f"  daylight evaporative fraction {ef:.2f}"
              f"  peak PBL {pbl[first].max():.0f} m")

    print(f"--- five-day drift ({args.case_prefix}) ---")
    for site, label, _ in SITES:
        ds = outputs[site]
        air = np.asarray(ds["T"].squeeze().values)[:, 0]
        lhf = np.asarray(ds["lhf"].squeeze().values)
        shf = np.asarray(ds["shf"].squeeze().values)
        first_day = air[SPINUP_STEPS:24].mean()
        last_day = air[96:120].mean()
        print(f"{site}: lowest-level air T {first_day:.1f} -> {last_day:.1f} K "
              f"({last_day - first_day:+.1f} K over five days), "
              f"peak SHF day 1 {shf[SPINUP_STEPS:24].max():.0f} -> "
              f"day 5 {shf[96:120].max():.0f} W m-2")

if __name__ == "__main__":
    main()
