import numpy as np
import xarray as xr
import metpy.calc as mpcalc
import metpy.constants
from metpy.units import units


#: Gravitational acceleration (m s-2) and dry-air specific heat (J kg-1 K-1).
_GRAVITY = 9.80665
_CP_DRY = 1004.6

#: ERA5 archives radiative fluxes as hourly accumulations in J m-2.
_SECONDS_PER_ACCUMULATION = 3600.0


def _as_flux_rate(da):
    """Return a radiative flux as a rate in W m-2.

    The downloader already divides the accumulated forecast stream by its
    accumulation period, but files produced by the older CDS-based workflow
    carry raw hourly accumulations in J m-2.  Both reach
    :func:`calculate_radiative_heating`, which needs W m-2, so the accumulated
    form is normalised here rather than at each call site.
    """
    units = str(da.attrs.get("units", "")).replace(" ", "").replace("*", "")
    if units in ("Jm-2", "Jm^-2", "Jm**-2"):
        return da / _SECONDS_PER_ACCUMULATION
    return da


def calculate_radiative_heating(swnet_top, swnet_sfc, lwnet_top, lwnet_sfc,
                                pressure, surface_pressure=None):
    """
    Diagnose a column-mean radiative heating rate profile (K/s).

    The net radiative flux convergence of the whole atmospheric column is

        Q = (swnet_top + lwnet_top) - (swnet_sfc + lwnet_sfc)      [W m-2]

    using ERA5's sign convention, in which all four terms are positive
    downward.  Distributing that convergence uniformly in mass over the column
    gives

        dT/dt = g * Q / (c_p * (p_surface - p_top))                [K s-1]

    Parameters
    ----------
    swnet_top, swnet_sfc, lwnet_top, lwnet_sfc : array-like, shape (time,)
        Net shortwave and longwave fluxes at the top of the atmosphere and at
        the surface, in W m-2 (ERA5 ``tsr``, ``ssr``, ``ttr``, ``str``).
    pressure : array-like, shape (levels,)
        Pressure levels in Pa.
    surface_pressure : array-like, shape (time,), optional
        Surface pressure in Pa.  Defaults to the highest pressure level.

    Returns
    -------
    numpy.ndarray, shape (time, levels)

    Notes
    -----
    This is a bulk column diagnostic, not a resolved heating profile: ERA5's
    archived fluxes are boundary values only, so the vertical structure of the
    heating cannot be recovered from them.  It is a fallback for the legacy
    grouped output format.  DEPHY cases written by this tool set
    ``radiation = "on"``, which makes the SCM compute radiation internally and
    ignore any prescribed rate.
    """
    pressure = np.asarray(pressure, dtype=float)
    swnet_top = np.asarray(swnet_top, dtype=float)
    swnet_sfc = np.asarray(swnet_sfc, dtype=float)
    lwnet_top = np.asarray(lwnet_top, dtype=float)
    lwnet_sfc = np.asarray(lwnet_sfc, dtype=float)

    column_convergence = (swnet_top + lwnet_top) - (swnet_sfc + lwnet_sfc)

    p_top = float(pressure.min())
    if surface_pressure is None:
        p_sfc = np.full_like(column_convergence, float(pressure.max()))
    else:
        p_sfc = np.asarray(surface_pressure, dtype=float)

    depth = np.maximum(p_sfc - p_top, 1.0)  # guard against a degenerate column
    heating = _GRAVITY * column_convergence / (_CP_DRY * depth)  # (time,)

    return np.repeat(heating[:, None], pressure.shape[0], axis=1)


def calculate_grid_spacing(lats, lons, center_idx=1):
    """
    Calculate grid spacing in meters, accounting for latitude.

    Parameters
    ----------
    lats : array-like, shape (3,) — latitude values for 3 grid points
    lons : array-like, shape (3,) — longitude values for 3 grid points
    center_idx : int — index of the center point (default=1)

    Returns
    -------
    dx : float — grid spacing in x-direction (metres) at center latitude
    dy : float — grid spacing in y-direction (metres)
    """
    R_EARTH = 6371000
    center_lat = lats[center_idx]
    lat_rad = np.deg2rad(center_lat)
    dy = R_EARTH * np.deg2rad(np.abs(lats[2] - lats[0])) / 2
    dx = R_EARTH * np.cos(lat_rad) * np.deg2rad(np.abs(lons[2] - lons[0])) / 2
    return dx, dy


def calculate_horizontal_gradients(var, lats, lons, center_idx=1):
    """
    Calculate horizontal gradients using centered differences accounting for latitude.

    Parameters
    ----------
    var : xarray.DataArray, shape (time, level, 3, 3)
    lats : array-like, shape (3,) — ERA5 convention: descending (north first)
    lons : array-like, shape (3,)
    center_idx : int — index of center point (default=1)

    Returns
    -------
    d_dx : xarray.DataArray, shape (time, level) — zonal gradient at center
    d_dy : xarray.DataArray, shape (time, level) — meridional gradient at center
    """
    dx, dy = calculate_grid_spacing(lats, lons, center_idx)
    d_dx = (var.isel(latitude=1, longitude=2, drop=True) - var.isel(latitude=1, longitude=0, drop=True)) / (2 * dx)
    # ERA5 latitudes are descending (north first): index 0 is northernmost.
    # The conventional northward gradient is (north - south) / (2*dy).
    d_dy = (var.isel(latitude=0, longitude=1, drop=True) - var.isel(latitude=2, longitude=1, drop=True)) / (2 * dy)
    return d_dx, d_dy


def calculate_geostrophic_wind(z, lat, lon):
    """
    Calculate geostrophic wind components from geopotential height.

    Parameters
    ----------
    z : xarray.DataArray, shape (time, level, 3, 3) — geopotential height
    lat : array-like, shape (3,) — latitudes of grid points
    lon : array-like, shape (3,) — longitudes of grid points

    Returns
    -------
    u_g : xarray.DataArray, shape (time, level) — zonal geostrophic wind
    v_g : xarray.DataArray, shape (time, level) — meridional geostrophic wind
    """
    f = 2 * 7.2921e-5 * np.sin(np.deg2rad(lat[1]))
    if np.isclose(f, 0.0):
        raise ValueError("Coriolis parameter is too close to zero at this latitude.")
    dz_dx, dz_dy = calculate_horizontal_gradients(z, lat, lon)
    u_g = -(1 / f) * 9.81 * dz_dy
    v_g =  (1 / f) * 9.81 * dz_dx
    return u_g, v_g


def theta_from_t(temperature, pressure):
    """Convert temperature to potential temperature."""
    return mpcalc.potential_temperature(
        pressure * units.Pa,
        temperature * units.kelvin
    ).metpy.dequantify()


def calculate_thetal(theta, t, q):
    """
    Calculate liquid water equivalent potential temperature (K).

    Parameters
    ----------
    theta : xarray.DataArray — potential temperature in K (from theta_from_t)
    t : xarray.DataArray — temperature in K
    q : xarray.DataArray — specific humidity in kg/kg

    Returns
    -------
    thetal : xarray.DataArray in K
    """
    t_qty = t * units.kelvin
    cpd = metpy.constants.Cp_d
    Lu = mpcalc.water_latent_heat_vaporization(t_qty)
    rl = mpcalc.mixing_ratio_from_specific_humidity(q * units("kg/kg"))
    thetal = theta - (theta / t_qty) * (Lu * rl / cpd)
    return thetal.metpy.dequantify()


def rho_from_sp(sp, t):
    """Calculate air density from surface pressure and temperature."""
    return sp / (287.05 * t)


def _three_point_first_derivative(f0, f1, f2, x0, x1, x2):
    """
    Three-point first derivative at x1 for non-uniform spacing.

    h1 = x1 - x0, h2 = x2 - x1 must both be strictly positive.
    f0, f1, f2 may be arrays of identical shape; x0, x1, x2 are scalars.

    Coefficients:
      c0 = -h2 / (h1*(h1 + h2))
      c1 =  (h2 - h1) / (h1*h2)
      c2 =  h1 / (h2*(h1 + h2))
    """
    h1 = x1 - x0
    h2 = x2 - x1
    if not (h1 > 0 and h2 > 0):
        raise ValueError("x0 < x1 < x2 must hold (strictly increasing coordinate).")
    c0 = -h2 / (h1 * (h1 + h2))
    c1 = (h2 - h1) / (h1 * h2)
    c2 =  h1 / (h2 * (h1 + h2))
    return c0 * f0 + c1 * f1 + c2 * f2


def calculate_advection(var, u, v, omega, lats, lons, pressure, earth_radius=6_371_000.0):
    """
    Calculate horizontal and vertical advection at the central grid cell (1, 1)
    from a 3x3 lat-lon patch using spherical metric factors.

    Parameters
    ----------
    var, u, v, omega : numpy arrays, shape (time, level, lat=3, lon=3)
        omega is the pressure vertical velocity [Pa/s].
    lats : 1-D array, shape (3,), degrees — ERA5 descending (north first)
    lons : 1-D array, shape (3,), degrees — increasing
    pressure : 1-D array, shape (level,), Pa — monotonically increasing
    earth_radius : float, metres

    Returns
    -------
    h_advec : numpy array, shape (time, level) — -(u*dvar/dx + v*dvar/dy)
    v_advec : numpy array, shape (time, level) — -(omega*dvar/dp)
    """
    nt, nz = var.shape[0], var.shape[1]

    # Longitude: convert to arc-length (metres), unwrap to avoid 0/360 discontinuity
    lam = np.deg2rad(lons.astype(float))
    lam_c = lam[1]
    lam = np.unwrap(lam - lam_c) + lam_c

    # Latitude: sort to strictly increasing order (south-to-north) for metric coords
    phi = np.deg2rad(lats.astype(float))
    lat_order = np.argsort(phi)
    phi_sorted = phi[lat_order]
    phi_c = phi[1]
    cos_phi_c = np.cos(phi_c)
    if np.abs(cos_phi_c) < 1e-6:
        raise ValueError("cos(phi_c) near zero: polar regions not supported.")

    s_lon = earth_radius * cos_phi_c * lam    # zonal arc-length, increasing
    s_lat = earth_radius * phi_sorted          # meridional arc-length, increasing
    x0, x1, x2 = s_lon[0], s_lon[1], s_lon[2]
    y0, y1, y2 = s_lat[0], s_lat[1], s_lat[2]

    # Zonal derivative: hold lat=center (index 1), vary lon
    f_lon0 = var[:, :, 1, 0]
    f_lon1 = var[:, :, 1, 1]
    f_lon2 = var[:, :, 1, 2]

    # Meridional derivative: hold lon=center (index 1), vary lat (sorted south→north)
    f_lat0 = var[:, :, lat_order[0], 1]
    f_lat1 = var[:, :, lat_order[1], 1]
    f_lat2 = var[:, :, lat_order[2], 1]

    dvardx = _three_point_first_derivative(f_lon0, f_lon1, f_lon2, x0, x1, x2)
    dvardy = _three_point_first_derivative(f_lat0, f_lat1, f_lat2, y0, y1, y2)

    # Vertical derivative in pressure coordinates at center point
    var_c = var[:, :, 1, 1]
    p = np.asarray(pressure, dtype=float)
    dvar_dp = np.empty_like(var_c)
    for t_i in range(nt):
        dvar_dp[t_i, :] = np.gradient(var_c[t_i, :], p, edge_order=2)

    u_c  = u[:, :, 1, 1]
    v_c  = v[:, :, 1, 1]
    om_c = omega[:, :, 1, 1]

    h_advec = -(u_c * dvardx + v_c * dvardy)
    v_advec = -(om_c * dvar_dp)
    return h_advec, v_advec


def era5_to_scm_forcing(ds):
    # Convert pressure coordinate to Pa if provided in hPa.  Scaling a
    # coordinate DataArray rescales its values but leaves the original hPa
    # coordinate attached, so it has to be re-indexed onto its own new values.
    # Without this, every later operation that mixes `pressure_levels` with a
    # variable off `ds` aligns two disjoint level coordinates and silently
    # produces a padded outer join.
    pressure_levels = ds.levels
    if float(pressure_levels.max()) < 2000.0:
        pressure_levels = pressure_levels * 100.0
    pressure_levels = pressure_levels.assign_coords(levels=pressure_levels.values)
    pressure_levels.attrs["units"] = "Pa"
    ds = ds.assign_coords(levels=pressure_levels)

    out = xr.Dataset(coords={"time": ds.time, "levels": pressure_levels})

    # Geopotential height from ERA5 geopotential
    z_units = ds.z.attrs.get("units", "m^2 s^-2")
    try:
        geopotential_height = mpcalc.geopotential_to_height(ds.z * units(z_units)).metpy.dequantify()
    except Exception:
        geopotential_height = ds.z / 9.80665
    u_g, v_g = calculate_geostrophic_wind(geopotential_height, ds.latitude.values, ds.longitude.values)
    zh_center = geopotential_height.isel(latitude=1, longitude=1, drop=True)  # (time, levels)

    # Convert omega (Pa/s) to w (m/s) for the w_ls output variable
    omega_center = ds.w.isel(latitude=1, longitude=1, drop=True) * units.Pa / units.second
    temperature_center = ds.t.isel(latitude=1, longitude=1, drop=True) * units.kelvin
    q_center = ds.q.isel(latitude=1, longitude=1, drop=True) * units("kg/kg")
    mixing_ratio = mpcalc.mixing_ratio_from_specific_humidity(q_center)
    pressure_2d = xr.DataArray(
        np.tile(pressure_levels.values, (ds.sizes["time"], 1)),
        dims=("time", "levels"),
        coords={"time": ds.time, "levels": pressure_levels},
    ) * units.Pa
    w_m_s = mpcalc.vertical_velocity(omega_center, pressure_2d, temperature_center, mixing_ratio).metpy.dequantify()

    # Prepare 4D numpy arrays (time, levels, latitude, longitude) for advection
    u_4d     = ds.u.transpose("time", "levels", "latitude", "longitude").values
    v_4d     = ds.v.transpose("time", "levels", "latitude", "longitude").values
    omega_4d = ds.w.transpose("time", "levels", "latitude", "longitude").values  # Pa/s

    # Potential temperature (4D) for advection
    theta_4d = theta_from_t(ds.t, pressure_levels).transpose("time", "levels", "latitude", "longitude").values

    h_advec_theta_np, v_advec_theta_np = calculate_advection(
        theta_4d, u_4d, v_4d, omega_4d,
        ds["latitude"].values,
        ds["longitude"].values,
        pressure_levels.values,
    )
    h_advec_qt_np, v_advec_qt_np = calculate_advection(
        ds.q.transpose("time", "levels", "latitude", "longitude").values,
        u_4d, v_4d, omega_4d,
        ds["latitude"].values,
        ds["longitude"].values,
        pressure_levels.values,
    )

    # Wrap numpy advection results back into xarray DataArrays
    def _make_da(arr):
        return xr.DataArray(arr, dims=("time", "levels"),
                            coords={"time": ds.time, "levels": pressure_levels})

    h_advec_theta = _make_da(h_advec_theta_np)
    v_advec_theta = _make_da(v_advec_theta_np)
    h_advec_qt    = _make_da(h_advec_qt_np)
    v_advec_qt    = _make_da(v_advec_qt_np)

    # Radiative tendency.  Note the bracket indexing for "str": ds.str would
    # resolve to xarray's string accessor, not the surface net thermal flux.
    dT_dt_rad = np.zeros((ds.sizes["time"], ds.sizes["levels"]))
    if all(vname in ds.variables for vname in ("tsr", "ssr", "ttr", "str")):
        def _flux(name):
            return _as_flux_rate(ds[name].isel(latitude=1, longitude=1)).values

        dT_dt_rad = calculate_radiative_heating(
            _flux("tsr"),
            _flux("ssr"),
            _flux("ttr"),
            _flux("str"),
            pressure_levels.values,
            surface_pressure=ds["sp"].isel(latitude=1, longitude=1).values,
        )

    # Liquid water potential temperature at center for thil_nudge
    theta_center = theta_from_t(ds.t.isel(latitude=1, longitude=1), pressure_levels)
    thetal_center = calculate_thetal(
        theta_center,
        ds.t.isel(latitude=1, longitude=1),
        ds.q.isel(latitude=1, longitude=1),
    )

    # Required forcing fields
    out["w_ls"]           = w_m_s.transpose("levels", "time")
    out["omega"]          = ds.w.isel(latitude=1, longitude=1).transpose("levels", "time")
    out["u_nudge"]        = ds.u.isel(latitude=1, longitude=1).transpose("levels", "time")
    out["v_nudge"]        = ds.v.isel(latitude=1, longitude=1).transpose("levels", "time")
    out["T_nudge"]        = ds.t.isel(latitude=1, longitude=1).transpose("levels", "time")
    out["thil_nudge"]     = thetal_center.transpose("levels", "time")
    out["qt_nudge"]       = ds.q.isel(latitude=1, longitude=1).transpose("levels", "time")
    out["u_g"]            = u_g.transpose("levels", "time")
    out["v_g"]            = v_g.transpose("levels", "time")
    out["h_advec_thetail"] = h_advec_theta.transpose("levels", "time")
    out["v_advec_thetail"] = v_advec_theta.transpose("levels", "time")
    out["h_advec_qt"]     = h_advec_qt.transpose("levels", "time")
    out["v_advec_qt"]     = v_advec_qt.transpose("levels", "time")
    out["dT_dt_rad"]      = xr.DataArray(
        dT_dt_rad.T,
        dims=("levels", "time"),
        coords={"levels": pressure_levels, "time": ds.time},
    )
    out["zh"]        = zh_center.transpose("levels", "time")
    out["p_surf"]    = ds.sp.isel(latitude=1, longitude=1)
    out["T_surf"]    = ds.t2m.isel(latitude=1, longitude=1)
    out["latitude"]  = xr.DataArray(float(ds.latitude.values[1]))
    out["longitude"] = xr.DataArray(float(ds.longitude.values[1]))

    # Variable attributes
    var_attrs = {
        'zh':                {'units': 'm',              'long_name': 'geopotential height'},
        'p_surf':            {'units': 'Pa',             'long_name': 'surface pressure'},
        'T_surf':            {'units': 'K',              'long_name': 'surface absolute temperature'},
        'w_ls':              {'units': 'm s^-1',         'long_name': 'large scale vertical velocity'},
        'omega':             {'units': 'Pa s^-1',        'long_name': 'large scale pressure vertical velocity'},
        'u_g':               {'units': 'm s^-1',         'long_name': 'large scale geostrophic E-W wind'},
        'v_g':               {'units': 'm s^-1',         'long_name': 'large scale geostrophic N-S wind'},
        'u_nudge':           {'units': 'm s^-1',         'long_name': 'E-W wind to nudge toward'},
        'v_nudge':           {'units': 'm s^-1',         'long_name': 'N-S wind to nudge toward'},
        'T_nudge':           {'units': 'K',              'long_name': 'absolute temperature to nudge toward'},
        'thil_nudge':        {'units': 'K',              'long_name': 'liquid water potential temperature to nudge toward'},
        'qt_nudge':          {'units': 'kg kg^-1',       'long_name': 'q_t to nudge toward'},
        'dT_dt_rad':         {'units': 'K s^-1',         'long_name': 'prescribed radiative heating rate'},
        'h_advec_thetail':   {'units': 'K s^-1',         'long_name': 'prescribed theta_il tendency due to horizontal advection'},
        'v_advec_thetail':   {'units': 'K s^-1',         'long_name': 'prescribed theta_il tendency due to vertical advection'},
        'h_advec_qt':        {'units': 'kg kg^-1 s^-1',  'long_name': 'prescribed q_t tendency due to horizontal advection'},
        'v_advec_qt':        {'units': 'kg kg^-1 s^-1',  'long_name': 'prescribed q_t tendency due to vertical advection'},
    }
    for var in out.variables:
        if var in var_attrs:
            out[var].attrs = var_attrs[var]

    return out
