"""
Convert ERA5-processed forcing data to CCPP-SCM DEPHY format.

The DEPHY format (version 1) is a flat NetCDF-4 file (no groups) with:
  - Dimensions: t0=1, time, lev, nsoil=4
  - Initial conditions at t0: thetal, qt, ua, va, pa, zh, ps, ql, qi, tke, o3,
                               stc, smc, slc
  - Time-varying forcing: ps_forc, pa_forc, zh_forc, tnthetal_adv, tnqt_adv,
                          wap, ug, vg
  - Scalar surface properties: slmsk, vegtyp, soiltyp, slopetyp, tsfco, vegfrac,
                                shdmin, shdmax, canopy, snowd, snoalb, tg3,
                                uustar, albedos, weasd, sncovr, tsfcl, roughness,
                                area, lat(time), lon(time)
  - Global attributes matching gabls3_noahmp style for a land+LSM case

Soil, snow, albedo and land-cover fields are taken from the ERA5 extraction
where it carries them, which it does whenever the download included the land
group.  Anything ERA5 cannot supply falls back to the gabls3_noahmp template,
and an explicit override supplied by the caller supersedes both.  Which field
came from where is recorded in the ``land_state_source`` global attribute.

Ozone is always seeded from the template, since ERA5's total column ozone
cannot be resolved into a profile.
"""

import json
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Optional, Union

import numpy as np
import xarray as xr

from . import templates
from .land_state import (
    IGBP_CLASS_NAMES,
    LAND_SCALAR_VARS,
    SOIL_PROFILE_VARS,
    STATSGO_CLASS_NAMES,
    parse_soil_type,
    parse_vegetation_type,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NOAH_SOIL_DEPTHS = np.array([0.1, 0.4, 1.0, 2.0])  # bottom of each layer (m)
_NSOIL = 4

#: Default pressure above which nudging is applied, Pa.  700 hPa sits above the
#: daytime boundary layer at most land sites, so the relaxation anchors the free
#: troposphere without constraining the surface coupling itself.
_DEFAULT_NUDGING_ABOVE_PA = 70000.0

#: Default relaxation timescale, s.  Three hours is the conventional choice and
#: is short enough to hold a month-long run without overwhelming the physics.
_DEFAULT_NUDGING_TIMESCALE_S = 10800.0

#: Named nudging configurations, and what each means in practice.
#:
#: A single-column model with prescribed forcing has nothing tying it to the
#: reanalysis it was built from, so errors accumulate.  Over a day or two that
#: does not matter.  Over weeks the column drifts and can destabilise, and the
#: choice below is a trade between that stability and how much of the
#: atmosphere is left free to respond to the surface.
NUDGING_PROFILES = {
    # Nothing constrains the column.  The honest configuration for a case of a
    # day or two, and what every earlier release of this package produced.
    "none": None,

    # Relax the free troposphere and leave the boundary layer free.  This is
    # the right choice for land-atmosphere coupling work, because the layer the
    # surface actually communicates with still evolves on its own.  It relies
    # on the low-level forcing being meaningful, which over pronounced relief
    # it is not: horizontal gradients taken on a pressure surface close to the
    # ground are contaminated by terrain, and the resulting spurious tendency
    # accumulates unchecked in exactly the layer this profile leaves free.
    # Pair it with an advection taper at rough sites.
    "free-troposphere": {
        "timescale_s": _DEFAULT_NUDGING_TIMESCALE_S,
        "above_pa": _DEFAULT_NUDGING_ABOVE_PA,
    },

    # Relax the whole column, boundary layer included.  The most robust option
    # and the one that survives complex terrain, at the cost of damping the
    # coupling signal: near-surface temperature and humidity are pulled toward
    # the reanalysis, so they no longer respond freely to the land surface.
    # Surface fluxes are still diagnosed from the land state and the soil still
    # evolves on its own, so the land initialisation remains visible, just
    # muted.
    "full-column": {
        "timescale_s": _DEFAULT_NUDGING_TIMESCALE_S,
        "above_pa": 0.0,
    },
}

#: DEPHY nudging arrays this writer emits, mapped to the intermediate field
#: each is built from.
#:
#: Temperature is nudged through ``ta_nud`` rather than ``thetal_nud`` even
#: though the case advects liquid water potential temperature.  CCPP-SCM
#: v7.0.0 resolves the cutoff level for ``pa_nudging_thetal`` by passing the
#: height profile to a routine that expects pressure (``scm_input.F90``, in the
#: ``nudging_thetal`` branch), so that path selects the wrong level.  The
#: ``ta`` branch passes the pressure profile and is correct.
_NUDGING_VARS = {
    "ua_nud": ("u_nudge", "m s-1", "eastward_wind_nudging"),
    "va_nud": ("v_nudge", "m s-1", "northward_wind_nudging"),
    "ta_nud": ("T_nudge", "K", "air_temperature_nudging"),
    "qt_nud": ("qt_nudge", "kg kg-1",
               "mass_fraction_of_water_in_air_nudging"),
}

#: The global attribute pairs that switch each array on.  DEPHY stores the
#: relaxation timescale itself in ``nudging_*``, not a boolean flag.
_NUDGING_ATTRS = {
    "ua_nud": ("nudging_ua", "pa_nudging_ua"),
    "va_nud": ("nudging_va", "pa_nudging_va"),
    "ta_nud": ("nudging_ta", "pa_nudging_ta"),
    "qt_nud": ("nudging_qt", "pa_nudging_qt"),
}

#: Short keys naming each nudged field in a per-field specification.
NUDGING_FIELDS = ("ua", "va", "ta", "qt")

#: Accepted spellings that are not the canonical key.  ``qv`` is offered because
#: moisture nudging is conventionally discussed in terms of vapour: this writer
#: emits ``qt_nud``/``nudging_qt``, and CCPP-SCM v7.0.0 handles the ``qt`` and
#: ``qv`` attributes identically (``scm_input.F90`` sets ``force_nudging_qv``
#: from either), so the distinction does not reach the model.
_NUDGING_FIELD_ALIASES = {"qv": "qt", "u": "ua", "v": "va", "t": "ta"}

_FIELD_TO_ARRAY = {field: f"{field}_nud" for field in NUDGING_FIELDS}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def convert_to_dephy(
    era5_processed: xr.Dataset,
    start_date: str,
    output_file: str,
    case_name: str,
    template_name: str = "gabls3",
    column_area: float = 145_000_000.0,
    scalars_override: Optional[dict] = None,
    nudging: str = "none",
    nudging_timescale_s: Optional[Union[float, Mapping]] = None,
    nudging_above_pa: Optional[Union[float, Mapping]] = None,
    advection_taper_pa: float = 0.0,
) -> None:
    """
    Write a DEPHY-format SCM driver file from processed ERA5 forcing data.

    Parameters
    ----------
    era5_processed : xr.Dataset
        Output of ``era5_to_scm_forcing()``.  Variables have dims
        ``(levels, time)``; surface scalars have dim ``(time,)``.
    start_date : str
        Simulation start as ``"YYYY-MM-DD"`` or ``"YYYY-MM-DD HH:MM:SS"``.
        Must match the first timestamp in ``era5_processed.time``.
    output_file : str
        Destination path for the ``*_SCM_driver.nc`` file.
    case_name : str
        Identifier used as the SCM ``case_name`` (matches the filename stem).
    template_name : str
        Key into ``templates.AVAILABLE_TEMPLATES``; used to seed static
        fields (soil, ozone, land-surface scalars).
    column_area : float
        Grid-cell area in m².  Default matches the GABLS3 configuration.
    scalars_override : dict, optional
        Mapping of scalar variable names to float values that override
        whatever the template provides (e.g. ``{'zorl': 10.0}``).
    nudging : {'none', 'free-troposphere', 'full-column'}
        Which named nudging configuration to use.  See
        :data:`NUDGING_PROFILES` for what each means in practice.  ``none``,
        the default, leaves the column free running, which is right for a case
        of a day or two and wrong for a month.
    nudging_timescale_s : float or mapping, optional
        Override the profile's relaxation timescale, in seconds.  Passing this
        while ``nudging`` is ``'none'`` turns nudging on with the default
        cutoff, which is how earlier versions of this function behaved.

        A mapping of field name to timescale sets each field separately, e.g.
        ``{'ua': 21600, 'va': 21600, 'ta': 21600, 'qt': 43200}``; fields left
        out fall back to the profile's timescale, and a field mapped to ``None``
        or ``0`` is left free.  Field names are those in
        :data:`NUDGING_FIELDS`, with ``qv`` accepted for ``qt``.

        Per-field control is what lets a study vary one field's constraint while
        holding the rest fixed — relaxing wind and temperature on a common
        timescale and moisture on its own, so that differences between runs are
        attributable to the moisture treatment alone.
    nudging_above_pa : float or mapping, optional
        Override the profile's cutoff pressure, in Pa.  Nudging is applied
        above this level, meaning at lower pressure.  Zero nudges everything.
        Accepts a per-field mapping on the same terms as
        ``nudging_timescale_s``.
    advection_taper_pa : float
        Depth above the surface, in Pa, over which the advective tendencies are
        smoothly reduced to zero.  Zero, the default, applies them unchanged.

        Horizontal gradients taken on a pressure surface that lies close to the
        ground are contaminated by terrain rather than describing advection,
        because the surface intersects the hillsides differently at each point
        of the stencil.  Over pronounced relief the resulting spurious tendency
        does not average out: at Walnut Gulch, where the surface varies by
        155 m across the stencil, it reaches a sustained 5 K per day of cooling
        in the lowest levels.  Tapering removes the forcing where it cannot be
        trusted and lets the boundary layer respond to the surface instead,
        which is what makes ``free-troposphere`` nudging usable at such a site.
        Around 15000 Pa is a reasonable starting point.
    """
    # ------------------------------------------------------------------
    # 1. Parse dates and build time coordinates
    # ------------------------------------------------------------------
    nudging_timescale_s, nudging_above_pa = _resolve_nudging(
        nudging, nudging_timescale_s, nudging_above_pa)
    if advection_taper_pa < 0:
        raise ValueError(
            f"advection_taper_pa must not be negative, got {advection_taper_pa}."
        )

    start_dt = _parse_date(start_date)
    start_date_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")

    era5_times = era5_processed.time.values  # numpy datetime64
    end_dt = _np64_to_datetime(era5_times[-1])
    end_date_str = end_dt.strftime("%Y-%m-%d %H:%M:%S")

    ref_np64 = np.datetime64(start_date_str.replace(" ", "T"), "s")
    time_vals = (era5_times - ref_np64).astype("timedelta64[s]").astype(np.float64)
    t0_val = np.array([0.0], dtype=np.float64)
    time_units = f"seconds since {start_date_str}"

    nt = len(time_vals)

    # ------------------------------------------------------------------
    # 2. Pressure levels (Pa, ordered surface first)
    # ------------------------------------------------------------------
    # DEPHY orders the vertical axis from the surface upward: every case in the
    # CCPP-SCM library, including the GABLS3 reference this package is modelled
    # on, has its highest pressure at index zero.  ERA5 hands back the opposite
    # order, so the whole dataset is reversed once here and every level indexed
    # array below inherits the corrected ordering.  Handing the SCM a top-first
    # profile does not fail loudly: it interpolates onto its own grid and
    # returns an isothermal free troposphere, which then drifts and eventually
    # destabilises the run.
    if era5_processed.levels.values[0] < era5_processed.levels.values[-1]:
        era5_processed = era5_processed.isel(
            levels=slice(None, None, -1)
        )

    # ERA5 reports every pressure level whether or not it lies underground, and
    # fills the buried ones by extrapolating downward from the free atmosphere.
    # At a high site that is a large share of the profile: six of the 37 levels
    # sit below the surface at Walnut Gulch, which is at about 1370 m.  Handing
    # those to the SCM puts fabricated air below the ground at the bottom of the
    # column, and no case in the CCPP-SCM library does that.  Levels that are
    # buried at any point in the run are dropped, which is the only choice that
    # keeps the vertical axis rectangular in time.
    surface_pressure = np.asarray(era5_processed["p_surf"].values, dtype=float)
    above_ground = era5_processed.levels.values <= surface_pressure.min()
    n_buried = int((~above_ground).sum())
    if n_buried:
        era5_processed = era5_processed.isel(levels=above_ground)
        print(f"  Dropped {n_buried} pressure level(s) below the surface "
              f"(surface pressure falls to {surface_pressure.min() / 100:.0f} hPa)")

    pressure_levels = era5_processed.levels.values.astype(np.float64)
    nlev = len(pressure_levels)
    if nlev < 2:
        raise ValueError(
            "Every ERA5 pressure level lies below the surface at this site; "
            "the case cannot be built."
        )

    # ------------------------------------------------------------------
    # 3. Load template for static fields (soil, ozone, land scalars)
    # ------------------------------------------------------------------
    template_file = templates.AVAILABLE_TEMPLATES[template_name]
    tpl_index = xr.open_dataset(template_file)
    tpl_initial = xr.open_dataset(template_file, group="initial")
    tpl_scalars = xr.open_dataset(template_file, group="scalars")

    # ------------------------------------------------------------------
    # 4. Dimension coordinates
    # ------------------------------------------------------------------
    coords = {
        "t0": xr.DataArray(
            t0_val, dims=["t0"],
            attrs={"units": time_units, "standard_name": "Initial time",
                   "calendar": "gregorian"},
        ),
        "time": xr.DataArray(
            time_vals, dims=["time"],
            attrs={"units": time_units, "standard_name": "Forcing time",
                   "calendar": "gregorian"},
        ),
        "lev": xr.DataArray(
            pressure_levels, dims=["lev"],
            attrs={"units": "Pa", "standard_name": "pressure"},
        ),
    }

    data_vars = {}

    # ------------------------------------------------------------------
    # 5. Soil-depth coordinate
    # ------------------------------------------------------------------
    data_vars["soil_depth"] = xr.DataArray(
        _NOAH_SOIL_DEPTHS.copy(), dims=["nsoil"],
        attrs={"units": "m", "standard_name": "depth of bottom of soil layers"},
    )

    # ------------------------------------------------------------------
    # 6. Spatial scalars broadcast over time (lat, lon)
    # ------------------------------------------------------------------
    lat_val = float(era5_processed["latitude"].values)
    lon_val = float(era5_processed["longitude"].values)
    data_vars["lat"] = xr.DataArray(
        np.full(nt, lat_val, dtype=np.float64), dims=["time"],
        attrs={"units": "degrees_north", "standard_name": "latitude"},
    )
    data_vars["lon"] = xr.DataArray(
        np.full(nt, lon_val, dtype=np.float64), dims=["time"],
        attrs={"units": "degrees_east", "standard_name": "longitude"},
    )

    # ------------------------------------------------------------------
    # 7. Land-surface scalars (seeded from template, overridden as needed)
    # ------------------------------------------------------------------
    _SCALAR_VARS = [
        "slmsk", "vegtyp", "soiltyp", "slopetyp", "tsfco", "vegfrac",
        "shdmin", "shdmax", "canopy", "hice", "fice", "tisfc", "snowd",
        "snoalb", "tg3", "uustar", "alvsf", "alnsf", "alvwf", "alnwf",
        "facsf", "facwf", "weasd", "sncovr", "tsfcl", "zorl", "zorll",
        "zorli", "zorlw",
    ]
    # A few scalars are spelled differently in the legacy grouped template
    # than in the DEPHY driver the SCM reads.
    _TEMPLATE_ALIASES = {"snowd": "snwdph"}

    for sv in _SCALAR_VARS:
        source_name = _TEMPLATE_ALIASES.get(sv, sv)
        if source_name in tpl_scalars:
            data_vars[sv] = tpl_scalars[source_name].astype(np.float64)

    # Precedence runs template < ERA5 < caller override.  Fields ERA5 cannot
    # supply keep the template value, so a partial land state is usable.
    provenance = _load_provenance(era5_processed)

    for sv in _SCALAR_VARS:
        if sv in LAND_SCALAR_VARS and sv in era5_processed:
            data_vars[sv] = xr.DataArray(
                float(era5_processed[sv].values),
                attrs=data_vars[sv].attrs if sv in data_vars else {},
            )

    if "tsfcl" not in era5_processed:
        # Without a land state there is no skin temperature to use, so the
        # 2 m temperature stands in as it did before the land group existed.
        t_surf_init = float(era5_processed["T_surf"].values[0])
        for skin_var in ("tsfco", "tsfcl"):
            if skin_var in data_vars:
                data_vars[skin_var] = xr.DataArray(
                    t_surf_init, attrs=data_vars[skin_var].attrs,
                )
                provenance[skin_var] = "ERA5 t2m (no skin temperature available)"

    # Apply any caller-supplied overrides
    if scalars_override:
        for sv, val in scalars_override.items():
            if sv in data_vars:
                data_vars[sv] = xr.DataArray(float(val), attrs=data_vars[sv].attrs)
                provenance[sv] = "user override"

    for sv in _SCALAR_VARS:
        provenance.setdefault(sv, f"{template_name} template")

    data_vars["area"] = xr.DataArray(
        float(column_area),
        attrs={"units": "m2", "standard_name": "grid cell area"},
    )

    # ------------------------------------------------------------------
    # 8. Helper closures
    # ------------------------------------------------------------------
    def _ic(varname):
        """First time step of a (levels, time) variable → (1, nlev) array."""
        return era5_processed[varname].isel(time=0).values.reshape(1, nlev).astype(np.float64)

    def _forc(varname):
        """Transpose (levels, time) → (time, lev) for forcing variables."""
        return era5_processed[varname].values.T.astype(np.float64)

    # ------------------------------------------------------------------
    # 9. Initial conditions at t0
    # ------------------------------------------------------------------
    data_vars["thetal"] = xr.DataArray(
        _ic("thil_nudge"), dims=["t0", "lev"],
        attrs={"units": "K", "standard_name": "air_liquid_potential_temperature"},
    )
    data_vars["qt"] = xr.DataArray(
        _ic("qt_nudge"), dims=["t0", "lev"],
        attrs={"units": "kg kg-1", "standard_name": "mass_fraction_of_water_in_air"},
    )
    data_vars["ua"] = xr.DataArray(
        _ic("u_nudge"), dims=["t0", "lev"],
        attrs={"units": "m s-1", "standard_name": "eastward_wind"},
    )
    data_vars["va"] = xr.DataArray(
        _ic("v_nudge"), dims=["t0", "lev"],
        attrs={"units": "m s-1", "standard_name": "northward_wind"},
    )
    data_vars["pa"] = xr.DataArray(
        pressure_levels.reshape(1, nlev), dims=["t0", "lev"],
        attrs={"units": "Pa", "standard_name": "air_pressure"},
    )
    data_vars["zh"] = xr.DataArray(
        _ic("zh"), dims=["t0", "lev"],
        attrs={"units": "m", "standard_name": "height"},
    )
    data_vars["ps"] = xr.DataArray(
        np.array([float(era5_processed["p_surf"].values[0])]),
        dims=["t0"],
        attrs={"units": "Pa", "standard_name": "surface_air_pressure"},
    )
    data_vars["ql"] = xr.DataArray(
        np.zeros((1, nlev), dtype=np.float64), dims=["t0", "lev"],
        attrs={"units": "kg kg-1",
               "standard_name": "mass_fraction_of_cloud_liquid_water_in_air"},
    )
    data_vars["qi"] = xr.DataArray(
        np.zeros((1, nlev), dtype=np.float64), dims=["t0", "lev"],
        attrs={"units": "kg kg-1",
               "standard_name": "mass_fraction_of_cloud_ice_water_in_air"},
    )
    data_vars["tke"] = xr.DataArray(
        np.zeros((1, nlev), dtype=np.float64), dims=["t0", "lev"],
        attrs={"units": "m2 s-2",
               "standard_name": "specific_turbulent_kinetic_energy"},
    )

    # Ozone: interpolate from template pressure levels to ERA5 levels
    data_vars["o3"] = xr.DataArray(
        _interp_ozone(tpl_index, tpl_initial, pressure_levels), dims=["t0", "lev"],
        attrs={"units": "kg kg-1",
               "standard_name": "mole_fraction_of_ozone_in_air"},
    )

    # Soil initial conditions: ERA5 where the extraction carries them, and the
    # template otherwise.
    for sv, sname, sunits in [
        ("stc", "initial profile of soil temperature", "K"),
        ("smc", "initial profile of soil moisture",    "m3 m-3"),
        ("slc", "initial profile of soil liquid moisture", "m3 m-3"),
    ]:
        if sv in era5_processed:
            vals = era5_processed[sv].values.reshape(1, _NSOIL).astype(np.float64)
        elif sv in tpl_initial:
            vals = tpl_initial[sv].values.reshape(1, _NSOIL).astype(np.float64)
            provenance[sv] = f"{template_name} template"
        else:
            vals = np.zeros((1, _NSOIL), dtype=np.float64)
            provenance[sv] = "zero fill"
        data_vars[sv] = xr.DataArray(
            vals, dims=["t0", "nsoil"],
            attrs={"units": sunits, "standard_name": sname},
        )

    # ------------------------------------------------------------------
    # 10. Forcing variables (time, lev)
    # ------------------------------------------------------------------
    data_vars["ps_forc"] = xr.DataArray(
        era5_processed["p_surf"].values.astype(np.float64), dims=["time"],
        attrs={"units": "Pa", "standard_name": "forcing_surface_air_pressure"},
    )
    data_vars["pa_forc"] = xr.DataArray(
        np.tile(pressure_levels, (nt, 1)), dims=["time", "lev"],
        attrs={"units": "Pa", "standard_name": "air_pressure_forcing"},
    )
    data_vars["zh_forc"] = xr.DataArray(
        _forc("zh"), dims=["time", "lev"],
        attrs={"units": "m", "standard_name": "height_forcing"},
    )
    # Large-scale forcing faded out near the ground when asked.  Both the
    # advective tendencies and the pressure velocity are tapered.  The
    # tendencies are computed from horizontal gradients on a pressure surface,
    # which terrain corrupts; the pressure velocity has to vanish at the ground
    # in any case, and ERA5 does not deliver that at a site where the lowest
    # levels sit within the relief.  Leaving it untapered was enough on its own
    # to cool a winter desert column by 13 K over a month.
    taper = advection_taper_weights(
        pressure_levels,
        np.asarray(era5_processed["p_surf"].values, dtype=float),
        advection_taper_pa,
    )
    if advection_taper_pa > 0:
        print(f"  Tapering advective tendencies to zero over the lowest "
              f"{advection_taper_pa / 100:.0f} hPa above the surface")

    data_vars["tnthetal_adv"] = xr.DataArray(
        (_forc("h_advec_thetail") + _forc("v_advec_thetail")) * taper,
        dims=["time", "lev"],
        attrs={
            "units": "K s-1",
            "standard_name": (
                "tendency_of_air_liquid_potential_temperature_due_to_advection"
            ),
        },
    )
    data_vars["tnqt_adv"] = xr.DataArray(
        (_forc("h_advec_qt") + _forc("v_advec_qt")) * taper,
        dims=["time", "lev"],
        attrs={
            "units": "kg kg-1 s-1",
            "standard_name": (
                "tendency_of_mass_fraction_of_water_in_air_due_to_advection"
            ),
        },
    )
    data_vars["wap"] = xr.DataArray(
        _forc("omega") * taper, dims=["time", "lev"],
        attrs={"units": "Pa s-1",
               "standard_name": "lagrangian_tendency_of_air_pressure"},
    )
    data_vars["ug"] = xr.DataArray(
        _forc("u_g"), dims=["time", "lev"],
        attrs={"units": "m s-1", "standard_name": "geostrophic_eastward_wind"},
    )
    data_vars["vg"] = xr.DataArray(
        _forc("v_g"), dims=["time", "lev"],
        attrs={"units": "m s-1", "standard_name": "geostrophic_northward_wind"},
    )

    # ------------------------------------------------------------------
    # 10b. Nudging targets, when the case is to be relaxed toward ERA5
    # ------------------------------------------------------------------
    nudged = []
    if nudging_timescale_s is not None:
        for field in nudging_timescale_s:
            name = _FIELD_TO_ARRAY[field]
            source, sunits, sname = _NUDGING_VARS[name]
            if source not in era5_processed:
                continue
            data_vars[name] = xr.DataArray(
                _forc(source), dims=["time", "lev"],
                attrs={"units": sunits, "standard_name": sname},
            )
            nudged.append(name)
        if not nudged:
            raise ValueError(
                "Nudging was requested but the processed forcings carry none "
                "of " + ", ".join(
                    _NUDGING_VARS[_FIELD_TO_ARRAY[f]][0] for f in nudging_timescale_s
                ) + "."
            )

    # ------------------------------------------------------------------
    # 11. Assemble dataset and set global attributes
    # ------------------------------------------------------------------
    ds_out = xr.Dataset(data_vars=data_vars, coords=coords)

    lat_str = f'{abs(lat_val):.4f}{"N" if lat_val >= 0 else "S"}'
    lon_str = f'{abs(lon_val):.4f}{"E" if lon_val >= 0 else "W"}'

    ds_out.attrs = {
        "description": f"Case data for {case_name} generated from ERA5 reanalysis",
        "missing_value": -9999.0,
        "case": f"{case_name}_{start_date_str}_{lon_str}{lat_str}",
        "title": f"Forcing and Initial Conditions for {case_name}",
        "reference": "ERA5 reanalysis (Copernicus/ECMWF) via era5-to-ccpp-scm-tool",
        "author": "era5-to-ccpp-scm-tool",
        "version": f"Created on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "format_version": "DEPHY SCM format version 1",
        "modifications": "",
        "script": "era5_to_ccpp_scm.to_dephy",
        "comment": f"Generated from ERA5 using {template_name} template for static fields",
        "start_date": start_date_str,
        "end_date": end_date_str,
        "forcing_scale": -1,
        # Advection flags
        "adv_ta": 0, "adv_qv": 0, "adv_ua": 0, "adv_va": 0,
        "adv_theta": 0, "adv_thetal": 1, "adv_qt": 1,
        "adv_rv": 0, "adv_rt": 0,
        # Radiation
        "radiation": "on",
        # Forcing methods
        "forc_wap": 1, "forc_wa": 0, "forc_geo": 1,
        # Nudging (all off)
        "nudging_ua": 0, "nudging_va": 0, "nudging_ta": 0,
        "nudging_theta": 0, "nudging_thetal": 0, "nudging_qv": 0,
        "nudging_qt": 0, "nudging_rv": 0, "nudging_rt": 0,
        "zh_nudging_ta": 0, "zh_nudging_theta": 0, "zh_nudging_thetal": 0,
        "zh_nudging_qv": 0, "zh_nudging_qt": 0, "zh_nudging_rv": 0,
        "zh_nudging_rt": 0, "zh_nudging_ua": 0, "zh_nudging_va": 0,
        "pa_nudging_ta": 0, "pa_nudging_theta": 0, "pa_nudging_thetal": 0,
        "pa_nudging_qv": 0, "pa_nudging_qt": 0, "pa_nudging_rv": 0,
        "pa_nudging_rt": 0, "pa_nudging_ua": 0, "pa_nudging_va": 0,
        # Surface / LSM
        "surface_type": "land",
        "surface_forcing_temp": "none",
        "surface_forcing_moisture": "none",
        "surface_forcing_wind": "none",
        "surface_forcing_lsm": "lsm",
        # Which land surface fields are site-derived and which are inherited.
        "land_state_source": json.dumps(provenance, sort_keys=True),
    }

    # DEPHY carries the relaxation timescale in the attribute itself, so
    # switching nudging on means replacing the zeros written above.
    for name in nudged:
        field = name.removesuffix("_nud")
        timescale_attr, pressure_attr = _NUDGING_ATTRS[name]
        ds_out.attrs[timescale_attr] = float(nudging_timescale_s[field])
        ds_out.attrs[pressure_attr] = float(nudging_above_pa[field])

    # ------------------------------------------------------------------
    # 12. Write with double precision for all float arrays
    # ------------------------------------------------------------------
    encoding = {
        v: {"dtype": "float64"}
        for v in ds_out.data_vars
        if ds_out[v].dtype.kind == "f"
    }
    # Also encode coordinates as double
    for c in ds_out.coords:
        if ds_out.coords[c].dtype.kind == "f":
            encoding[c] = {"dtype": "float64"}

    ds_out.to_netcdf(output_file, encoding=encoding, format="NETCDF4_CLASSIC")
    print(f"  Wrote DEPHY file: {output_file}")

    if nudged:
        summary = ", ".join(
            f"{field} {nudging_timescale_s[field]:.0f} s above "
            f"{nudging_above_pa[field] / 100.0:.0f} hPa"
            for field in sorted(name.removesuffix("_nud") for name in nudged)
        )
        free = sorted(set(NUDGING_FIELDS) - {n.removesuffix("_nud") for n in nudged})
        print(f"  Nudging toward ERA5: {summary}"
              + (f"; free: {', '.join(free)}" if free else ""))

    inherited = sorted(k for k, v in provenance.items() if "template" in v)
    if inherited:
        print(f"  Land fields inherited from the {template_name} template: "
              f"{', '.join(inherited)}")


def write_case_namelist(
    case_name: str,
    output_file: str,
    sfc_roughness_length_cm: float = 15.0,
    column_area: float = 145_000_000.0,
    reference_profile_choice: int = 2,
    do_spinup: bool = True,
    spinup_timesteps: int = 12,
) -> None:
    """
    Write the SCM case configuration namelist for a DEPHY land+LSM case.

    Parameters
    ----------
    case_name : str
        Must match the stem of the ``*_SCM_driver.nc`` file.
    output_file : str
        Destination path for the ``.nml`` file.
    sfc_roughness_length_cm : float
        Surface roughness length in cm; should match ``zorl`` in the DEPHY file.
    column_area : float
        Horizontal grid-cell area in m².
    reference_profile_choice : int
        1 = tropical McClatchy, 2 = mid-latitude summer standard atmosphere.
    do_spinup : bool
        Whether to run a spinup period before the main simulation.
    spinup_timesteps : int
        Number of timesteps in the spinup.
    """
    spinup_str = ".true." if do_spinup else ".false."
    content = (
        f"&case_config\n"
        f"    case_name = '{case_name}'\n"
        f"    column_area = {column_area:.1f}\n"
        f"    do_spinup = {spinup_str}\n"
        f"    spinup_timesteps = {spinup_timesteps}\n"
        f"    sfc_roughness_length_cm = {sfc_roughness_length_cm:.4f}\n"
        f"    reference_profile_choice = {reference_profile_choice}\n"
        f"    input_type = 1\n"
        f"    lsm_ics = .true.\n"
        f"/\n"
    )
    with open(output_file, "w") as fh:
        fh.write(content)
    print(f"  Wrote namelist:   {output_file}")


#: Fields :func:`override_land_state` will patch, and how each is interpreted.
#: Vegetation and soil types accept class names and FLUXNET codes as well as
#: numbers; everything else is a plain float.
_OVERRIDABLE = {
    "vegtyp": parse_vegetation_type,
    "soiltyp": parse_soil_type,
    "vegfrac": float,
    "shdmin": float,
    "shdmax": float,
    "slopetyp": float,
    "zorl": float,
    "zorll": float,
    "snoalb": float,
    "tg3": float,
    "canopy": float,
    "facsf": float,
    "facwf": float,
}


def override_land_state(driver_file: str, output_file: Optional[str] = None,
                        **overrides) -> dict:
    """Patch land surface descriptors in a finished DEPHY driver file.

    ERA5 describes land cover and soil texture at 0.25 degrees, which is a
    coarse descriptor of an eddy covariance footprint.  Where a site publishes
    its own classification, this replaces the derived value without rerunning
    the download and conversion, which for a long case is by far the expensive
    part.

    Vegetation type accepts an IGBP class number, an IGBP class name, or a
    FLUXNET/AmeriFlux abbreviation such as ``GRA`` or ``DBF``, so a site's
    published descriptor can be used directly.  Soil type accepts a STATSGO
    class number or name.

    Parameters
    ----------
    driver_file : str
        Path to an existing ``*_SCM_driver.nc``.
    output_file : str, optional
        Where to write the patched file.  Defaults to editing in place.
    **overrides
        Any of the fields in :data:`_OVERRIDABLE`.

    Returns
    -------
    dict
        The values actually written, after parsing.

    Examples
    --------
    Pin a case to the IGBP class published for the flux tower::

        override_land_state("fluxnet_US-Whs_SCM_driver.nc", vegtyp="OSH")
    """
    unknown = sorted(set(overrides) - set(_OVERRIDABLE))
    if unknown:
        raise ValueError(
            f"Cannot override {', '.join(unknown)}; this function handles "
            f"{', '.join(sorted(_OVERRIDABLE))}."
        )
    if not overrides:
        raise ValueError("No overrides given.")

    with xr.open_dataset(driver_file) as opened:
        ds = opened.load()

    provenance = _load_provenance(ds)
    applied = {}

    for name, raw in overrides.items():
        if raw is None:
            continue
        value = _OVERRIDABLE[name](raw)
        attrs = ds[name].attrs if name in ds else {}
        ds[name] = xr.DataArray(np.float64(value), attrs=attrs)
        provenance[name] = "user override"
        applied[name] = value

    ds.attrs["land_state_source"] = json.dumps(provenance, sort_keys=True)
    ds.attrs["modifications"] = _append_modification(
        ds.attrs.get("modifications", ""),
        "land surface overrides: "
        + ", ".join(f"{k}={v}" for k, v in sorted(applied.items())),
    )

    destination = output_file or driver_file
    encoding = {v: {"dtype": "float64"} for v in ds.data_vars
                if ds[v].dtype.kind == "f"}
    ds.to_netcdf(destination, encoding=encoding, format="NETCDF4_CLASSIC")

    for name, value in sorted(applied.items()):
        label = ""
        if name == "vegtyp":
            label = f" ({IGBP_CLASS_NAMES[int(value)]})"
        elif name == "soiltyp":
            label = f" ({STATSGO_CLASS_NAMES[int(value)]})"
        print(f"  {name} -> {value}{label}")
    print(f"  Wrote {destination}")

    return applied


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _normalise_field_spec(value, what):
    """Return ``(scalar, per_field)`` for a scalar-or-mapping nudging argument.

    Exactly one of the two is not ``None``.  Mapping keys are canonicalised
    through :data:`_NUDGING_FIELD_ALIASES`; unknown keys raise rather than being
    silently dropped, because a typo would otherwise leave a field un-nudged
    without saying so.
    """
    if value is None:
        return None, None
    if isinstance(value, Mapping):
        resolved = {}
        for key, item in value.items():
            canonical = _NUDGING_FIELD_ALIASES.get(str(key).lower(), str(key).lower())
            if canonical not in NUDGING_FIELDS:
                raise ValueError(
                    f"Unknown nudging field {key!r} in {what}; expected one of "
                    f"{', '.join(NUDGING_FIELDS)} "
                    f"(aliases: {', '.join(sorted(_NUDGING_FIELD_ALIASES))})."
                )
            resolved[canonical] = None if item is None else float(item)
        return None, resolved
    return float(value), None


def _resolve_nudging(profile, timescale_override, pressure_override):
    """Turn a named profile plus overrides into per-field timescales and cutoffs.

    Both overrides accept either a scalar, which applies to every field, or a
    mapping of field name to value, which lets one field be relaxed on its own
    schedule or switched off entirely (``None`` or ``0``).  Per-field control is
    what makes a moisture-sensitivity design expressible: hold the wind and
    temperature anchored to the reanalysis on a fixed timescale and vary only
    how tightly humidity is held, so any difference between runs is attributable
    to moisture rather than to the whole column being constrained differently.

    Returns
    -------
    (timescales, pressures) : tuple of dict or (None, None)
        Dicts keyed by the field names in :data:`NUDGING_FIELDS`, holding only
        the fields that are actually nudged.  ``(None, None)`` when nudging is
        off entirely.
    """
    if profile not in NUDGING_PROFILES:
        raise ValueError(
            f"Unknown nudging profile {profile!r}; expected one of "
            f"{', '.join(sorted(NUDGING_PROFILES))}."
        )

    ts_scalar, ts_fields = _normalise_field_spec(timescale_override, "nudging_timescale_s")
    pa_scalar, pa_fields = _normalise_field_spec(pressure_override, "nudging_above_pa")

    # A bare non-positive timescale is a caller error rather than a request:
    # "switch this off" is only meaningful per field, where it names which one.
    if ts_scalar is not None and ts_scalar <= 0:
        raise ValueError(
            f"The nudging timescale must be positive, got {ts_scalar}. To leave "
            "a field free, map it to None in a per-field specification, or pass "
            "nudging='none' to disable nudging altogether."
        )

    settings = NUDGING_PROFILES[profile]
    if settings is None:
        if ts_scalar is None and ts_fields is None:
            return None, None
        # A bare timescale with no profile is how this was configured before
        # the profiles existed, so it keeps working.  A per-field mapping with
        # no profile turns on exactly the fields it names.
        settings = {"timescale_s": ts_scalar, "above_pa": _DEFAULT_NUDGING_ABOVE_PA}

    timescales, pressures = {}, {}
    for field in NUDGING_FIELDS:
        if ts_fields is not None:
            timescale = ts_fields.get(field, settings["timescale_s"])
        elif ts_scalar is not None:
            timescale = ts_scalar
        else:
            timescale = settings["timescale_s"]

        # None or a non-positive value means "leave this field free".
        if timescale is None or timescale == 0:
            continue
        if timescale < 0:
            raise ValueError(
                f"The nudging timescale must be positive, got {timescale} for {field!r}."
            )

        if pa_fields is not None:
            pressure = pa_fields.get(field, settings["above_pa"])
        elif pa_scalar is not None:
            pressure = pa_scalar
        else:
            pressure = settings["above_pa"]

        timescales[field] = float(timescale)
        pressures[field] = float(0.0 if pressure is None else pressure)

    if not timescales:
        raise ValueError(
            "Nudging was requested but every field was switched off; pass "
            "nudging='none' to disable nudging, or give at least one field a "
            "positive timescale."
        )
    return timescales, pressures


def parse_nudging_spec(value):
    """Parse a command-line nudging value into a float or a per-field mapping.

    Accepts a bare number (``"21600"``) or a comma-separated field list
    (``"ua=21600,va=21600,ta=21600,qt=43200"``).  ``off`` and ``none`` switch a
    field off.  Returns ``None`` for an empty value.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    if "=" not in text:
        return float(text)

    spec = {}
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(
                f"Malformed per-field nudging entry {chunk!r}; expected 'field=value'."
            )
        key, _, item = chunk.partition("=")
        item = item.strip().lower()
        spec[key.strip()] = None if item in ("off", "none", "") else float(item)
    return spec


def advection_taper_weights(pressure_levels, surface_pressure, taper_pa):
    """Weights that fade the advective tendencies out toward the ground.

    The weight is zero at the surface and one at ``taper_pa`` above it, joined
    by a smoothstep so the forcing has no kink where the taper ends.  The
    surface pressure varies through the run, so the weights do too.

    Parameters
    ----------
    pressure_levels : array_like, shape (nlev,)
        Case pressure levels in Pa.
    surface_pressure : array_like, shape (ntime,)
        Surface pressure in Pa at each forcing time.
    taper_pa : float
        Depth of the taper in Pa.  Zero returns weights of one throughout.

    Returns
    -------
    numpy.ndarray, shape (ntime, nlev)
    """
    pressure_levels = np.asarray(pressure_levels, dtype=float)
    surface_pressure = np.asarray(surface_pressure, dtype=float)

    if taper_pa <= 0:
        return np.ones((surface_pressure.size, pressure_levels.size))

    height_above_surface = surface_pressure[:, None] - pressure_levels[None, :]
    fraction = np.clip(height_above_surface / taper_pa, 0.0, 1.0)
    return fraction * fraction * (3.0 - 2.0 * fraction)


def _load_provenance(source) -> dict:
    """Read the land state provenance record, tolerating its absence."""
    raw = source.attrs.get("land_state_source") if hasattr(source, "attrs") else None
    if not raw:
        return {}
    try:
        loaded = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _append_modification(existing: str, note: str) -> str:
    """Append a note to the DEPHY ``modifications`` attribute."""
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"{stamp}: {note}"
    return f"{existing}; {entry}" if existing else entry


def _parse_date(date_str: str) -> datetime:
    """Accept 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'."""
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse date string: {date_str!r}")


def _np64_to_datetime(np64) -> datetime:
    """Convert a numpy datetime64 scalar to a naive UTC Python datetime."""
    ts = (np64 - np.datetime64("1970-01-01T00:00:00")) / np.timedelta64(1, "s")
    return datetime.fromtimestamp(float(ts), timezone.utc).replace(tzinfo=None)


def _interp_ozone(tpl_index: xr.Dataset, tpl_initial: xr.Dataset,
                  target_levels: np.ndarray) -> np.ndarray:
    """
    Interpolate the ozone profile from template pressure levels to
    the ERA5 pressure levels, returning shape (1, nlev).
    """
    if "ozone" not in tpl_initial:
        return np.zeros((1, len(target_levels)), dtype=np.float64)

    tpl_lev = tpl_index["levels"].values.astype(np.float64)
    tpl_o3  = tpl_initial["ozone"].values.astype(np.float64)

    # Ensure both arrays are sorted ascending (low-to-high pressure)
    sort_idx = np.argsort(tpl_lev)
    tpl_lev_sorted = tpl_lev[sort_idx]
    tpl_o3_sorted  = tpl_o3[sort_idx]

    target_sorted_idx = np.argsort(target_levels)
    target_sorted     = target_levels[target_sorted_idx]

    o3_sorted = np.interp(target_sorted, tpl_lev_sorted, tpl_o3_sorted)

    # Un-sort back to original target_levels order
    o3 = np.empty_like(o3_sorted)
    o3[target_sorted_idx] = o3_sorted

    return o3.reshape(1, len(target_levels))
