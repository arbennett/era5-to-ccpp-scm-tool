"""
Extract ERA5 columns from the NSF NCAR ERA5 archive.

This replaces the previous Copernicus CDS (``cdsapi``) downloader.  The NSF
NCAR archive needs no account, no API key and no request queue: the data sit
in a public S3 bucket and are read directly over HTTP range requests, so a
three-by-three stencil is pulled without staging the global field.

Two backends expose the identical directory layout:

  ``aws``    anonymous ``s3://nsf-ncar-era5`` — works anywhere
  ``glade``  ``/glade/campaign/collections/rda/data/d633000`` — NCAR-only,
             much faster because it is a POSIX read

``source="auto"`` (the default) picks ``glade`` when that path is readable and
falls back to ``aws`` otherwise.

The output files carry CDS-style lowercase variable names (``t``, ``q``, ``sp``,
``t2m`` …) and the dimensions ``(time, levels, latitude, longitude)``, so the
downstream conversion code is unchanged from the CDS era.
"""

from __future__ import annotations

import datetime as dt
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Sequence

import numpy as np
import xarray as xr

from .era5_catalog import (
    GLADE_ROOT,
    INVARIANT_VARIABLES,
    INVARIANT_YYYYMM,
    LAND_VARIABLES,
    PRESSURE_VARIABLES,
    RADIATION_VARIABLES,
    S3_BUCKET,
    SURFACE_VARIABLES,
    Era5Variable,
    months_between,
    parse_span,
)

# ERA5 is on a regular 0.25-degree latitude/longitude grid running from 90N to
# 90S and from 0E to 359.75E.
_GRID_STEP = 0.25
_N_LAT = 721
_N_LON = 1440

#: Accumulated fluxes are archived as hourly totals in J m-2.
_SECONDS_PER_ACCUMULATION = 3600.0

_DEFAULT_MAX_WORKERS = 8


# ---------------------------------------------------------------------------
# Archive backends
# ---------------------------------------------------------------------------

class _Archive:
    """Common interface over the S3 and Glade copies of ERA5."""

    def listdir(self, stream: str, yyyymm: str) -> Sequence[str]:
        raise NotImplementedError

    def open(self, stream: str, yyyymm: str, name: str):
        raise NotImplementedError


class _GladeArchive(_Archive):
    def __init__(self, root: str = GLADE_ROOT):
        self.root = root

    def listdir(self, stream, yyyymm):
        path = os.path.join(self.root, stream, yyyymm)
        if not os.path.isdir(path):
            return []
        return sorted(os.listdir(path))

    def open(self, stream, yyyymm, name):
        return os.path.join(self.root, stream, yyyymm, name)


class _S3Archive(_Archive):
    def __init__(self, bucket: str = S3_BUCKET):
        import s3fs

        self.bucket = bucket
        # A single anonymous filesystem is shared by all worker threads; s3fs
        # is thread-safe and pools connections internally.
        self.fs = s3fs.S3FileSystem(anon=True)

    def listdir(self, stream, yyyymm):
        prefix = f"{self.bucket}/{stream}/{yyyymm}"
        try:
            return sorted(os.path.basename(k) for k in self.fs.ls(prefix))
        except FileNotFoundError:
            return []

    def open(self, stream, yyyymm, name):
        return self.fs.open(f"{self.bucket}/{stream}/{yyyymm}/{name}", "rb")


def _make_archive(source: str) -> _Archive:
    if source == "auto":
        source = "glade" if os.path.isdir(GLADE_ROOT) else "aws"
    if source == "glade":
        if not os.path.isdir(GLADE_ROOT):
            raise RuntimeError(
                f"source='glade' requested but {GLADE_ROOT} is not readable. "
                "Use source='aws' to read from S3 instead."
            )
        return _GladeArchive()
    if source == "aws":
        return _S3Archive()
    raise ValueError(f"Unknown source {source!r}; expected 'auto', 'aws' or 'glade'.")


# ---------------------------------------------------------------------------
# Date and grid helpers
# ---------------------------------------------------------------------------

def _parse_date(value, end_of_day: bool = False) -> dt.datetime:
    """Parse ``YYYY-MM-DD`` or ``YYYY-MM-DD HH:MM:SS`` into a datetime.

    A bare date used as a range end means "through the end of that day", which
    matches how the CDS-based downloader behaved.
    """
    if isinstance(value, dt.datetime):
        return value
    if isinstance(value, dt.date):
        return dt.datetime(value.year, value.month, value.day,
                           23 if end_of_day else 0)
    text = str(value).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M",
                "%Y-%m-%d"):
        try:
            parsed = dt.datetime.strptime(text, fmt)
        except ValueError:
            continue
        if fmt == "%Y-%m-%d" and end_of_day:
            parsed = parsed.replace(hour=23)
        return parsed
    raise ValueError(f"Cannot parse date string: {value!r}")


def _stencil_indices(lat: float, lon: float, step: int = 1):
    """Return the 3x3 grid indices centred on ``(lat, lon)``.

    ``step`` widens the stencil in whole grid cells while keeping it 3x3; the
    conversion code differences the outer points, so a larger step trades
    spatial resolution for a better-conditioned gradient.

    Longitudes wrap across the prime meridian.  Latitudes are clamped so the
    stencil stays on the grid at the poles.
    """
    if step < 1:
        raise ValueError("step must be a positive number of grid cells.")
    if not -90.0 <= lat <= 90.0:
        raise ValueError(f"Latitude {lat} is outside [-90, 90].")

    # Latitude index 0 is 90N and increases southward.
    lat_centre = int(round((90.0 - lat) / _GRID_STEP))
    lat_centre = min(max(lat_centre, step), _N_LAT - 1 - step)
    lat_idx = [lat_centre - step, lat_centre, lat_centre + step]

    lon_centre = int(round((lon % 360.0) / _GRID_STEP)) % _N_LON
    lon_idx = [(lon_centre + offset * step) % _N_LON for offset in (-1, 0, 1)]

    return lat_idx, lon_idx


def _unwrap_longitude(da: xr.DataArray) -> xr.DataArray:
    """Give a stencil a monotonic longitude coordinate.

    Selecting across the prime meridian yields longitudes like ``[359.75, 0,
    0.25]``.  Downstream gradient code differences the outer points, so the
    coordinate is unwrapped to ``[-0.25, 0, 0.25]`` to keep the spacing right.
    """
    lons = np.asarray(da["longitude"].values, dtype=float)
    unwrapped = np.rad2deg(np.unwrap(np.deg2rad(lons)))
    return da.assign_coords(longitude=unwrapped)


def _read_all_stencils(field: xr.DataArray, stencils) -> Dict[str, xr.DataArray]:
    """Read every site's 3x3 stencil from an open ``field``.

    Sites are read one at a time.  Batching them into a single strided read was
    measurably slower: h5py degrades sharply once a selection is fancy-indexed
    along more than a couple of points per axis, and the wider selection lost
    more than the repeated chunk decompression cost.

    ERA5 stores one whole global field per compressed chunk, so the expensive
    part — fetching and inflating the chunk — is shared across sites by virtue
    of reading them from the same open file.
    """
    return {
        site: _unwrap_longitude(
            field.isel(latitude=lat_idx, longitude=lon_idx).load()
        )
        for site, (lat_idx, lon_idx) in stencils.items()
    }


def _select_files(archive: _Archive, variable: Era5Variable,
                  start: dt.datetime, end: dt.datetime):
    """List the archive files for ``variable`` overlapping ``[start, end]``.

    Returns ``(yyyymm, filename)`` pairs in chronological order.  The
    accumulated-flux stream stores each half-month under its starting month,
    including the portion that spills into the next month, so the month scan is
    widened by a day at each edge.
    """
    token = f"{variable.stream}.{variable.code}.{variable.grid}."
    scan_start = start - dt.timedelta(days=1)
    scan_end = end + dt.timedelta(days=1)

    selected = []
    for yyyymm in months_between(scan_start, scan_end):
        for name in archive.listdir(variable.stream, yyyymm):
            if not name.startswith(token):
                continue
            span = parse_span(name)
            if span is None:
                continue
            file_start, file_end = span
            if file_end >= start and file_start <= end:
                selected.append((yyyymm, name, file_start))

    selected.sort(key=lambda item: item[2])
    # A file can be reachable from more than one month scan; keep the first.
    seen = set()
    unique = []
    for yyyymm, name, _ in selected:
        if name in seen:
            continue
        seen.add(name)
        unique.append((yyyymm, name))
    return unique


# ---------------------------------------------------------------------------
# Per-variable readers
# ---------------------------------------------------------------------------

def _open_dataset(archive: _Archive, stream: str, yyyymm: str, name: str):
    handle = archive.open(stream, yyyymm, name)
    engine = "h5netcdf" if not isinstance(handle, str) else None
    return xr.open_dataset(handle, engine=engine)


def _missing_files_error(short_name, variable, start, end):
    return FileNotFoundError(
        f"No ERA5 files found for {short_name!r} ({variable.code}) "
        f"between {start} and {end}."
    )


def _read_instantaneous(archive, short_name, variable, start, end,
                        stencils) -> Dict[str, xr.DataArray]:
    """Read an analysis variable for every requested site.

    ``stencils`` maps a site key to its ``(lat_idx, lon_idx)`` pair.  All sites
    are pulled from the same open file: ERA5 stores one whole global field per
    chunk, so the cost is dominated by opening the file at all, and additional
    sites are close to free.
    """
    files = _select_files(archive, variable, start, end)
    if not files:
        raise _missing_files_error(short_name, variable, start, end)

    pieces = {site: [] for site in stencils}
    for yyyymm, name in files:
        with _open_dataset(archive, variable.stream, yyyymm, name) as ds:
            field = ds[variable.nc_name].sel(time=slice(start, end))
            for site, da in _read_all_stencils(field, stencils).items():
                pieces[site].append(da)

    out = {}
    for site, parts in pieces.items():
        combined = xr.concat(parts, dim="time") if len(parts) > 1 else parts[0]
        out[site] = combined.sortby("time").rename(short_name)
    return out


def _read_accumulated(archive, short_name, variable, start, end,
                      stencils) -> Dict[str, xr.DataArray]:
    """Read an accumulated flux for every site, as a mean rate in W m-2.

    The forecast stream is dimensioned ``(forecast_initial_time,
    forecast_hour)``; ERA5 initialises at 06 and 18 UTC and accumulates over
    each of the following 12 hours.  Flattening those two axes onto the valid
    time gives a continuous hourly series.
    """
    files = _select_files(archive, variable, start, end)
    if not files:
        raise _missing_files_error(short_name, variable, start, end)

    pieces = {site: [] for site in stencils}
    for yyyymm, name in files:
        with _open_dataset(archive, variable.stream, yyyymm, name) as ds:
            per_site = _read_all_stencils(ds[variable.nc_name], stencils)
            for site, da in per_site.items():
                init = da["forecast_initial_time"].values
                hours = da["forecast_hour"].values.astype("timedelta64[h]")
                valid = (init[:, None] + hours[None, :]).ravel()

                da = da.stack(time=("forecast_initial_time", "forecast_hour"))
                da = da.drop_vars(
                    [c for c in ("time", "forecast_initial_time",
                                 "forecast_hour") if c in da.coords]
                )
                da = da.assign_coords(time=("time", valid))
                pieces[site].append(da.transpose("time", ...))

    out = {}
    for site, parts in pieces.items():
        combined = xr.concat(parts, dim="time") if len(parts) > 1 else parts[0]
        combined = combined.sortby("time")
        # Overlapping half-month files repeat the boundary hour.
        _, keep = np.unique(combined["time"].values, return_index=True)
        combined = combined.isel(time=keep).sel(time=slice(start, end))

        combined = combined / _SECONDS_PER_ACCUMULATION
        combined.attrs["units"] = "W m**-2"
        combined.attrs["comment"] = (
            "hourly accumulation from the ERA5 forecast stream, divided by 3600 s"
        )
        out[site] = combined.rename(short_name)
    return out


def _read_invariant(archive, short_name, variable, start, end,
                    stencils) -> Dict[str, xr.DataArray]:
    """Read a time-invariant field for every site.

    The invariant stream holds a single file per variable covering the whole
    record, so ``start`` and ``end`` are accepted for signature compatibility
    with the other readers and are not used to select it.  The degenerate
    length-one time dimension is dropped.
    """
    token = f"{variable.stream}.{variable.code}.{variable.grid}."
    names = [n for n in archive.listdir(variable.stream, INVARIANT_YYYYMM)
             if n.startswith(token)]
    if not names:
        raise _missing_files_error(short_name, variable, start, end)

    with _open_dataset(archive, variable.stream, INVARIANT_YYYYMM,
                       sorted(names)[0]) as ds:
        field = ds[variable.nc_name]
        out = {}
        for site, da in _read_all_stencils(field, stencils).items():
            if "time" in da.dims:
                da = da.isel(time=0, drop=True)
            out[site] = da.rename(short_name)
    return out


#: Dispatch table for :func:`_worker`; keys are passed as the ``kind`` argument.
_READERS = {
    "instantaneous": _read_instantaneous,
    "accumulated": _read_accumulated,
    "invariant": _read_invariant,
}

#: One archive handle per worker process, built lazily on first use.
_PROCESS_ARCHIVE = {}


def _worker(job):
    """Read one variable inside a worker process.

    Runs in its own interpreter, which is what makes concurrency safe here:
    the HDF5 library beneath both NetCDF engines is not thread-safe, so
    concurrent opens in a single process segfault.
    """
    source, kind, short_name, variable, start, end, stencils = job
    archive = _PROCESS_ARCHIVE.get(source)
    if archive is None:
        archive = _PROCESS_ARCHIVE[source] = _make_archive(source)
    return _READERS[kind](archive, short_name, variable, start, end, stencils)


def _fetch_group(source: str, variables: Dict[str, Era5Variable], start, end,
                 stencils, kind: str, max_workers: int) -> Dict[str, xr.Dataset]:
    """Read a group of variables concurrently, returning one Dataset per site.

    Work is split by variable rather than by site, because every site is served
    from the same file read.
    """
    jobs = [
        (source, kind, short_name, variable, start, end, stencils)
        for short_name, variable in variables.items()
    ]

    if max_workers <= 1 or len(jobs) == 1:
        results = [_worker(job) for job in jobs]
    else:
        # "spawn" keeps a forked copy of an already-initialised HDF5 library
        # out of the children, which is a known source of crashes.
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=min(max_workers, len(jobs)),
                                 mp_context=context) as pool:
            results = list(pool.map(_worker, jobs))

    return {
        site: xr.merge([r[site] for r in results],
                       combine_attrs="drop_conflicts")
        for site in stencils
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def download_era5_sites(
    sites,
    start_date,
    end_date,
    output_dir: str,
    source: str = "auto",
    stencil_step: int = 1,
    include_radiation: bool = True,
    include_land: bool = True,
    max_workers: int = _DEFAULT_MAX_WORKERS,
    per_site_subdir: bool = True,
):
    """Extract ERA5 column stencils for many sites in a single pass.

    ERA5 is chunked one whole global field per time step, so reading a 3x3
    stencil costs the same as reading the entire file — roughly 1.3 GB per
    pressure-level variable per day.  That cost is per *file*, not per site, so
    extracting fifty sites together costs about the same as extracting one.
    Prefer this over calling :func:`download_era5_time_series` in a loop.

    Parameters
    ----------
    sites : mapping or iterable
        Either ``{site_id: (lat, lon)}`` or an iterable of
        ``(site_id, lat, lon)`` triples.  Longitudes may be -180..180 or
        0..360.
    start_date, end_date : str or datetime
        ``YYYY-MM-DD`` or ``YYYY-MM-DD HH:MM:SS``.  A bare end date means the
        end of that day.
    output_dir : str
        Root directory for output; created if absent.
    source : {'auto', 'aws', 'glade'}
        Which copy of the archive to read.  ``auto`` prefers Glade when the
        NCAR mount is visible and otherwise reads from the public S3 bucket.
    stencil_step : int
        Stencil half-width in grid cells.  ``1`` gives the native 0.25-degree
        spacing; larger values widen the 3x3 stencil, which damps noise in the
        computed horizontal gradients.
    include_radiation : bool
        Also fetch accumulated radiative fluxes from the forecast stream.
    include_land : bool
        Also fetch the soil, snow, albedo and land-cover fields needed to
        initialise the land surface model from ERA5 rather than from the
        packaged template.  The soil and snow fields join the surface file;
        the land-cover and soil-texture classes are written to a third,
        time-invariant file.
    max_workers : int
        Number of worker processes; work is divided by variable.
    per_site_subdir : bool
        Write each site into ``{output_dir}/{site_id}/``.  When False, all
        files land in ``output_dir`` with the site id as the filename stem.

    Returns
    -------
    dict
        ``{site_id: (pressure_level_path, surface_path)}`` when ``include_land``
        is False, otherwise ``{site_id: (pressure_level_path, surface_path,
        invariant_path)}``.
    """
    stencil_coords = _normalise_sites(sites)
    if not stencil_coords:
        raise ValueError("No sites given.")

    start = _parse_date(start_date)
    end = _parse_date(end_date, end_of_day=True)
    if end < start:
        raise ValueError(f"end_date ({end}) precedes start_date ({start}).")

    # Resolve 'auto' once so every worker process reads the same copy.
    if source == "auto":
        source = "glade" if os.path.isdir(GLADE_ROOT) else "aws"
    _make_archive(source)  # fail fast on a bad source or a missing s3fs
    backend = "Glade" if source == "glade" else "AWS S3"

    stencils = {
        site: _stencil_indices(lat, lon, step=stencil_step)
        for site, (lat, lon) in stencil_coords.items()
    }

    print(f"Reading ERA5 from {backend} for {len(stencils)} site(s), "
          f"{start:%Y-%m-%d %H:%M} to {end:%Y-%m-%d %H:%M}")

    print(f"  pressure levels: {', '.join(sorted(PRESSURE_VARIABLES))}")
    pl_by_site = _fetch_group(source, PRESSURE_VARIABLES, start, end,
                              stencils, "instantaneous", max_workers)

    print(f"  surface: {', '.join(sorted(SURFACE_VARIABLES))}")
    sfc_by_site = _fetch_group(source, SURFACE_VARIABLES, start, end,
                               stencils, "instantaneous", max_workers)

    rad_by_site = {}
    if include_radiation:
        print(f"  radiation: {', '.join(sorted(RADIATION_VARIABLES))}")
        rad_by_site = _fetch_group(source, RADIATION_VARIABLES, start, end,
                                   stencils, "accumulated", max_workers)

    land_by_site = {}
    inv_by_site = {}
    if include_land:
        print(f"  land surface: {', '.join(sorted(LAND_VARIABLES))}")
        land_by_site = _fetch_group(source, LAND_VARIABLES, start, end,
                                    stencils, "instantaneous", max_workers)
        print(f"  invariant: {', '.join(sorted(INVARIANT_VARIABLES))}")
        inv_by_site = _fetch_group(source, INVARIANT_VARIABLES, start, end,
                                   stencils, "invariant", max_workers)

    written = {}
    for site, (lat, lon) in stencil_coords.items():
        pl = pl_by_site[site]
        pl = pl.rename({"level": "levels"}) if "level" in pl.dims else pl
        sfc = sfc_by_site[site]

        if site in rad_by_site:
            # Radiative fluxes are hourly accumulations valid over the
            # preceding hour and can begin one step later than the analysis;
            # align them onto the analysis times.
            rad = rad_by_site[site].reindex(
                time=sfc["time"], method="nearest",
                tolerance=np.timedelta64(1, "h"),
            )
            sfc = xr.merge([sfc, rad], combine_attrs="drop_conflicts")

        if site in land_by_site:
            # The land fields come off the same analysis stream on the same
            # times, so they merge into the surface file without alignment.
            sfc = xr.merge([sfc, land_by_site[site]],
                           combine_attrs="drop_conflicts")

        provenance = {
            "source": f"NSF NCAR ERA5 (RDA ds633.0) via {backend}",
            "source_url": "https://registry.opendata.aws/nsf-ncar-era5/",
            "site_id": str(site),
            "site_latitude": float(lat),
            "site_longitude": float(lon),
            "stencil_step_cells": int(stencil_step),
            "created_by": "era5-to-ccpp-scm-tool",
            "created_on": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        pl.attrs.update(provenance)
        sfc.attrs.update(provenance)

        site_dir = os.path.join(output_dir, str(site)) if per_site_subdir \
            else output_dir
        os.makedirs(site_dir, exist_ok=True)
        pl_path = os.path.join(site_dir, f"{site}_pl.nc")
        sfc_path = os.path.join(site_dir, f"{site}_sfc.nc")

        _drop_helper_coords(pl).to_netcdf(pl_path)
        _drop_helper_coords(sfc).to_netcdf(sfc_path)
        print(f"  wrote {pl_path}")
        print(f"  wrote {sfc_path}")

        if site in inv_by_site:
            inv = inv_by_site[site]
            inv.attrs.update(provenance)
            inv_path = os.path.join(site_dir, f"{site}_inv.nc")
            _drop_helper_coords(inv).to_netcdf(inv_path)
            print(f"  wrote {inv_path}")
            written[site] = (pl_path, sfc_path, inv_path)
        else:
            written[site] = (pl_path, sfc_path)

    return written


def download_era5_time_series(
    start_date,
    end_date,
    lat: float,
    lon: float,
    output_dir: str,
    name: str = "era5",
    source: str = "auto",
    stencil_step: int = 1,
    include_radiation: bool = True,
    include_land: bool = True,
    max_workers: int = _DEFAULT_MAX_WORKERS,
):
    """Extract a 3x3 ERA5 column stencil for one site.

    Writes ``{output_dir}/{name}_pl.nc`` and ``{output_dir}/{name}_sfc.nc``,
    plus ``{output_dir}/{name}_inv.nc`` when ``include_land`` is set, and
    returns those paths.  See :func:`download_era5_sites` for the argument
    meanings; for more than one site call that function directly, as it
    amortises the file reads across sites.
    """
    written = download_era5_sites(
        {name: (lat, lon)},
        start_date=start_date,
        end_date=end_date,
        output_dir=output_dir,
        source=source,
        stencil_step=stencil_step,
        include_radiation=include_radiation,
        include_land=include_land,
        max_workers=max_workers,
        per_site_subdir=False,
    )
    return written[name]


def _normalise_sites(sites) -> Dict[str, tuple]:
    """Accept either ``{id: (lat, lon)}`` or ``[(id, lat, lon), ...]``."""
    if hasattr(sites, "items"):
        return {str(k): (float(v[0]), float(v[1])) for k, v in sites.items()}
    out = {}
    for entry in sites:
        site_id, lat, lon = entry
        out[str(site_id)] = (float(lat), float(lon))
    return out


def _drop_helper_coords(ds: xr.Dataset) -> xr.Dataset:
    """Drop RDA bookkeeping variables that confuse the conversion step."""
    drop = [v for v in ("utc_date", "forecast_initial_time", "forecast_hour")
            if v in ds.variables]
    return ds.drop_vars(drop) if drop else ds
