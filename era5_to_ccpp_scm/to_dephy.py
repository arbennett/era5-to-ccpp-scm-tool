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

Static fields (soil, ozone, land-surface properties) are seeded from the
gabls3_noahmp template and can be overridden per site via the scalars dict.
"""

from datetime import datetime
from typing import Optional

import numpy as np
import xarray as xr

from . import templates


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NOAH_SOIL_DEPTHS = np.array([0.1, 0.4, 1.0, 2.0])  # bottom of each layer (m)
_NSOIL = 4


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
    """
    # ------------------------------------------------------------------
    # 1. Parse dates and build time coordinates
    # ------------------------------------------------------------------
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
    # 2. Pressure levels (Pa, ascending from model top to surface)
    # ------------------------------------------------------------------
    pressure_levels = era5_processed.levels.values.astype(np.float64)
    nlev = len(pressure_levels)

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

    # Override surface skin temperature with ERA5 value at t=0
    t_surf_init = float(era5_processed["T_surf"].values[0])
    for skin_var in ("tsfco", "tsfcl"):
        if skin_var in data_vars:
            data_vars[skin_var] = xr.DataArray(
                t_surf_init,
                attrs=data_vars[skin_var].attrs,
            )

    # Apply any caller-supplied overrides
    if scalars_override:
        for sv, val in scalars_override.items():
            if sv in data_vars:
                data_vars[sv] = xr.DataArray(float(val), attrs=data_vars[sv].attrs)

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

    # Soil initial conditions from template (Noah LSM, 4 layers)
    for sv, sname, sunits in [
        ("stc", "initial profile of soil temperature", "K"),
        ("smc", "initial profile of soil moisture",    "kg"),
        ("slc", "initial profile of soil liquid moisture", "kg"),
    ]:
        if sv in tpl_initial:
            vals = tpl_initial[sv].values.reshape(1, _NSOIL).astype(np.float64)
        else:
            vals = np.zeros((1, _NSOIL), dtype=np.float64)
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
    data_vars["tnthetal_adv"] = xr.DataArray(
        _forc("h_advec_thetail") + _forc("v_advec_thetail"),
        dims=["time", "lev"],
        attrs={
            "units": "K s-1",
            "standard_name": (
                "tendency_of_air_liquid_potential_temperature_due_to_advection"
            ),
        },
    )
    data_vars["tnqt_adv"] = xr.DataArray(
        _forc("h_advec_qt") + _forc("v_advec_qt"),
        dims=["time", "lev"],
        attrs={
            "units": "kg kg-1 s-1",
            "standard_name": (
                "tendency_of_mass_fraction_of_water_in_air_due_to_advection"
            ),
        },
    )
    data_vars["wap"] = xr.DataArray(
        _forc("omega"), dims=["time", "lev"],
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
    }

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


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _parse_date(date_str: str) -> datetime:
    """Accept 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'."""
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse date string: {date_str!r}")


def _np64_to_datetime(np64) -> datetime:
    """Convert a numpy datetime64 scalar to a Python datetime."""
    ts = (np64 - np.datetime64("1970-01-01T00:00:00")) / np.timedelta64(1, "s")
    return datetime.utcfromtimestamp(float(ts))


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
