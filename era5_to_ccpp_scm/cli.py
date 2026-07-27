import os
import shutil

import click
import xarray as xr
import numpy as np

from . import templates
from .util import _maybe_open
from typing import Union, Optional
from .download_era5 import download_era5_time_series
from .convert_forcing import era5_to_scm_forcing
from .to_dephy import convert_to_dephy, write_case_namelist


@click.group()
def cli():
    """
    ERA5 to CCPP-SCM data processing CLI. Commands are:
     1. download_era5
     2. convert_forcings
     3. convert_era5_from_template  (legacy grouped-format output)
     4. convert_to_dephy            (DEPHY format for direct SCM use)
     5. run_full_pipeline
    """
    pass


_SOURCE_HELP = ("Which copy of the NSF NCAR ERA5 archive to read: 'aws' for "
                "the public S3 bucket, 'glade' for the NCAR mount, or 'auto' "
                "to prefer Glade when it is visible.")


@cli.command(name='download_era5')
@click.option('--start_date', type=str, required=True)
@click.option('--end_date', type=str, required=True)
@click.option('--lat', type=float, required=True)
@click.option('--lon', type=float, required=True)
@click.option('--output_dir', type=str, default='.')
@click.option('--name', type=str, default='era5')
@click.option('--source', type=click.Choice(['auto', 'aws', 'glade']),
              default='auto', help=_SOURCE_HELP)
@click.option('--stencil_step', type=int, default=1,
              help='Stencil half-width in grid cells; larger values widen the '
                   '3x3 stencil and damp noise in horizontal gradients.')
@click.option('--no_radiation', is_flag=True, default=False,
              help='Skip the accumulated radiative fluxes.')
@click.option('--max_workers', type=int, default=8,
              help='Number of worker processes (work is split by variable).')
def download_era5(
    start_date: str,
    end_date: str,
    lat: float,
    lon: float,
    output_dir: str,
    name: str,
    source: str,
    stencil_step: int,
    no_radiation: bool,
    max_workers: int,
):
    """
    Download ERA5 data for a given time period and location.
    """
    download_era5_time_series(
        start_date=start_date,
        end_date=end_date,
        lat=lat,
        lon=lon,
        output_dir=output_dir,
        name=name,
        source=source,
        stencil_step=stencil_step,
        include_radiation=not no_radiation,
        max_workers=max_workers,
    )

def _core_convert_forcings(
    era5_surface_file: Union[str, xr.Dataset],
    era5_pressure_levels_file: Union[str, xr.Dataset],
    output_file: Optional[str]=None,
    era5_rad_file: Union[str, xr.Dataset, None]=None,
):
    """
    Merge raw ERA5 files and derive the SCM forcing fields.

    The downloader folds the radiative fluxes into the surface file, so
    ``era5_rad_file`` is only needed for older extractions that kept them in a
    third file.
    """
    # Datasets opened from a path are ours to close.  Reading them eagerly and
    # releasing the handles keeps the returned object independent of the files
    # it came from: otherwise the result stays lazily backed by open handles,
    # and a later open of the same path in the same process fails with an HDF
    # error.
    opened = []

    def _open(source):
        dataset = _maybe_open(source)
        if isinstance(source, str):
            opened.append(dataset)
        return dataset

    parts = [_open(era5_surface_file)]
    if era5_rad_file is not None:
        parts.append(_open(era5_rad_file))
    parts.append(_open(era5_pressure_levels_file))
    # The inputs hold disjoint variables, so no_conflicts is both the safe
    # choice and the one that flags an unexpected overlap instead of hiding it.
    try:
        ds = xr.merge(parts, compat="no_conflicts").load()
    finally:
        for dataset in opened:
            dataset.close()

    rename_map = {}
    if "valid_time" in ds.dims or "valid_time" in ds.coords:
        rename_map["valid_time"] = "time"
    # 'pressure_level' is the CDS name; 'level' is what the NSF NCAR archive
    # uses.  Files written by this tool already use 'levels'.
    for source_name in ("pressure_level", "level"):
        if source_name in ds.dims or source_name in ds.coords:
            rename_map[source_name] = "levels"
    if rename_map:
        ds = ds.rename(rename_map)

    out = era5_to_scm_forcing(ds)
    if output_file is not None:
        out.to_netcdf(output_file, format="NETCDF4")
    return out


@cli.command(name='convert_forcings')
@click.option('-s', '--era5_surface_file', type=str, required=True)
@click.option('-p', '--era5_pressure_levels_file', type=str, required=True)
@click.option('-o', '--output_file', type=str)
@click.option('-r', '--era5_rad_file', type=str, default=None,
              help='Optional separate radiative-flux file. Only needed for '
                   'older extractions; the downloader now writes the fluxes '
                   'into the surface file.')
def convert_forcings(
    era5_surface_file: Union[str, xr.Dataset],
    era5_pressure_levels_file: Union[str, xr.Dataset],
    output_file: Optional[str]=None,
    era5_rad_file: Union[str, xr.Dataset, None]=None,
):
    """
    Convert ERA5 data to intermediate SCM forcing file.
    """
    return _core_convert_forcings(era5_surface_file, era5_pressure_levels_file,
                                  output_file, era5_rad_file)


def _core_convert_era5_from_template(
    era5_processed_forcings: Union[str, xr.Dataset],
    template: str,
    output_file: str
):
    """Legacy grouped-format output (kept for backward compatibility)."""
    # Open the template file and the ERA5 data
    template_file = templates.AVAILABLE_TEMPLATES[template]
    template_index = xr.open_dataset(template_file)
    template_forcing = xr.open_dataset(template_file, group='forcing')
    template_initial = xr.open_dataset(template_file, group='initial')
    template_scalars = xr.open_dataset(template_file, group='scalars')
    era5_ds = _maybe_open(era5_processed_forcings)

    # Interpolate the ERA5 data to the template levels
    era5_ds = era5_ds.interp(levels=template_index.levels, method='linear')

    # Convert the timestamps to SCM format
    # And also convert the time variable in the index/root group
    dt = era5_ds.time.values[1] - era5_ds.time.values[0]
    dt = dt.astype('timedelta64[s]').astype(int)
    new_time = np.arange(0, era5_ds.time.size * dt, dt)
    era5_ds = era5_ds.assign_coords(time=new_time)
    era5_ds.time.attrs['units'] = 's'
    era5_ds.time.attrs['long_name'] = 'elapsed time since the beginning of the simulation'
    if "time" in template_index.data_vars:
        template_index = template_index.drop_vars("time")
    template_index = template_index.assign_coords({"time": new_time})

    # Build the forcing group on the *case's* time axis.  Copying the template
    # group and assigning into it aligns the template's own time coordinate
    # (145 steps for GABLS3) against the case's, which fails for every case
    # that is not exactly as long as the template.  The template contributes
    # variable names, dtypes and attributes; the values come from ERA5.
    forcing_vars = {}
    for var in template_forcing.data_vars:
        dims = template_forcing[var].dims
        if var in era5_ds and set(dims).issubset(set(era5_ds[var].dims)):
            values = era5_ds[var].transpose(*dims).values
        else:
            # Not derivable from ERA5; leave it zeroed on the case's axes
            # rather than carrying the template's unrelated time series.
            values = np.zeros(tuple(era5_ds.sizes[d] for d in dims))
        forcing_vars[var] = xr.DataArray(
            values.astype(template_forcing[var].dtype, copy=False),
            dims=dims,
            attrs=template_forcing[var].attrs,
        )

    forcing_out = xr.Dataset(
        forcing_vars,
        coords={"time": new_time, "levels": template_index["levels"]},
    )
    forcing_out.time.attrs = era5_ds.time.attrs

    # Update scalar lat/lon if present from processed forcing output.
    if "lat" in template_scalars and "latitude" in era5_ds:
        template_scalars["lat"] = xr.DataArray(float(era5_ds["latitude"].values))
    if "lon" in template_scalars and "longitude" in era5_ds:
        template_scalars["lon"] = xr.DataArray(float(era5_ds["longitude"].values))

    # Write groups
    template_index.to_netcdf(output_file, mode='w')
    forcing_out.to_netcdf(output_file, group='forcing', mode='a')
    template_initial.to_netcdf(output_file, group='initial', mode='a')
    template_scalars.to_netcdf(output_file, group='scalars', mode='a')


@cli.command(name='convert_era5_from_template')
@click.option('-f', '--era5_processed_forcings', type=str)
@click.option('-t', '--template', type=str, default='gabls3')
@click.option('-o', '--output_file', type=str)
def convert_era5_from_template(
    era5_processed_forcings: Union[str, xr.Dataset],
    template: str,
    output_file: str
):
    """
    Generate a legacy grouped-format CCPP-SCM input file (backward-compatible).
    Use convert_to_dephy for DEPHY format compatible directly with the SCM.
    """
    _core_convert_era5_from_template(era5_processed_forcings, template, output_file)


def _core_convert_to_dephy(
    era5_processed_forcings: Union[str, xr.Dataset],
    start_date: str,
    case_name: str,
    output_file: str,
    template: str = 'gabls3',
    column_area: float = 145_000_000.0,
    sfc_roughness_length_cm: float = 15.0,
    namelist_file: Optional[str] = None,
    scm_cases_dir: Optional[str] = None,
    scm_config_dir: Optional[str] = None,
):
    """
    Core logic: convert processed ERA5 forcings to DEPHY format and optionally
    deposit output files directly into an SCM directory tree.
    """
    era5_ds = _maybe_open(era5_processed_forcings)
    if isinstance(era5_processed_forcings, str):
        # Read it now and let go of the handle; see _core_convert_forcings.
        with era5_ds:
            era5_ds = era5_ds.load()

    convert_to_dephy(
        era5_processed=era5_ds,
        start_date=start_date,
        output_file=output_file,
        case_name=case_name,
        template_name=template,
        column_area=column_area,
        scalars_override={'zorl': sfc_roughness_length_cm,
                          'zorll': sfc_roughness_length_cm},
    )

    # Write companion .nml file
    if namelist_file is None:
        namelist_file = output_file.replace('_SCM_driver.nc', '.nml')
        if not namelist_file.endswith('.nml'):
            namelist_file = os.path.splitext(output_file)[0] + '.nml'

    write_case_namelist(
        case_name=case_name,
        output_file=namelist_file,
        sfc_roughness_length_cm=sfc_roughness_length_cm,
        column_area=column_area,
    )

    # Optionally install directly into the SCM directory tree
    if scm_cases_dir is not None:
        dest_nc = os.path.join(scm_cases_dir, f'{case_name}_SCM_driver.nc')
        shutil.copy2(output_file, dest_nc)
        print(f"  Installed case data: {dest_nc}")

    if scm_config_dir is not None:
        dest_nml = os.path.join(scm_config_dir, f'{case_name}.nml')
        shutil.copy2(namelist_file, dest_nml)
        print(f"  Installed namelist:  {dest_nml}")


@cli.command(name='convert_to_dephy')
@click.option('-f', '--era5_processed_forcings', type=str, required=True,
              help='Processed ERA5 forcing file (output of convert_forcings)')
@click.option('--start_date', type=str, required=True,
              help='Simulation start date YYYY-MM-DD or YYYY-MM-DD HH:MM:SS')
@click.option('--case_name', type=str, required=True,
              help='SCM case identifier (used in filename and namelist)')
@click.option('-o', '--output_file', type=str, required=True,
              help='Output *_SCM_driver.nc path')
@click.option('-t', '--template', type=str, default='gabls3',
              help='Template name for static fields (default: gabls3)')
@click.option('--column_area', type=float, default=145_000_000.0,
              help='Grid-cell area in m² (default: 145000000.0)')
@click.option('--sfc_roughness_length_cm', type=float, default=15.0,
              help='Surface roughness length in cm (default: 15.0)')
@click.option('--namelist_file', type=str, default=None,
              help='Path for output .nml file (default: adjacent to output_file)')
@click.option('--scm_cases_dir', type=str, default=None,
              help='If set, copy *_SCM_driver.nc here after writing')
@click.option('--scm_config_dir', type=str, default=None,
              help='If set, copy *.nml here after writing')
def convert_to_dephy_cmd(
    era5_processed_forcings: str,
    start_date: str,
    case_name: str,
    output_file: str,
    template: str,
    column_area: float,
    sfc_roughness_length_cm: float,
    namelist_file: Optional[str],
    scm_cases_dir: Optional[str],
    scm_config_dir: Optional[str],
):
    """
    Convert processed ERA5 forcings to DEPHY-format CCPP-SCM driver file.

    Also writes a companion case-config namelist (.nml) and can optionally
    install both files directly into an SCM directory tree.
    """
    _core_convert_to_dephy(
        era5_processed_forcings=era5_processed_forcings,
        start_date=start_date,
        case_name=case_name,
        output_file=output_file,
        template=template,
        column_area=column_area,
        sfc_roughness_length_cm=sfc_roughness_length_cm,
        namelist_file=namelist_file,
        scm_cases_dir=scm_cases_dir,
        scm_config_dir=scm_config_dir,
    )


@cli.command(name='run_full_pipeline')
@click.option('--start_date', type=str, required=True,
              help='Simulation start date YYYY-MM-DD or YYYY-MM-DD HH:MM:SS')
@click.option('--end_date', type=str, required=True,
              help='Simulation end date YYYY-MM-DD or YYYY-MM-DD HH:MM:SS')
@click.option('--lat', type=float, required=True)
@click.option('--lon', type=float, required=True)
@click.option('--case_name', type=str, required=True,
              help='SCM case identifier (used in filename and namelist)')
@click.option('--output_dir', type=str, default='.')
@click.option('--template', type=str, default='gabls3')
@click.option('--source', type=click.Choice(['auto', 'aws', 'glade']),
              default='auto', help=_SOURCE_HELP)
@click.option('--stencil_step', type=int, default=1,
              help='Stencil half-width in grid cells.')
@click.option('--max_workers', type=int, default=8,
              help='Number of worker processes for the download.')
@click.option('--column_area', type=float, default=145_000_000.0,
              help='Grid-cell area in m²')
@click.option('--sfc_roughness_length_cm', type=float, default=15.0,
              help='Surface roughness length in cm')
@click.option('--scm_cases_dir', type=str, default=None,
              help='If set, install *_SCM_driver.nc into this directory')
@click.option('--scm_config_dir', type=str, default=None,
              help='If set, install *.nml into this directory')
def run_full_pipeline(
    start_date: str,
    end_date: str,
    lat: float,
    lon: float,
    case_name: str,
    output_dir: str = '.',
    template: str = 'gabls3',
    source: str = 'auto',
    stencil_step: int = 1,
    max_workers: int = 8,
    column_area: float = 145_000_000.0,
    sfc_roughness_length_cm: float = 15.0,
    scm_cases_dir: Optional[str] = None,
    scm_config_dir: Optional[str] = None,
):
    """
    Full pipeline: download ERA5 → convert forcings → write DEPHY SCM driver.

    Writes into output_dir:
      {case_name}_pl.nc         — raw ERA5 pressure-level data
      {case_name}_sfc.nc        — raw ERA5 surface data
      {case_name}_SCM_driver.nc — DEPHY-format SCM input (ready for run_scm.py)
      {case_name}.nml           — SCM case-config namelist
    """
    os.makedirs(output_dir, exist_ok=True)

    era5_pl_file  = os.path.join(output_dir, f'{case_name}_pl.nc')
    era5_sfc_file = os.path.join(output_dir, f'{case_name}_sfc.nc')
    dephy_file    = os.path.join(output_dir, f'{case_name}_SCM_driver.nc')
    nml_file      = os.path.join(output_dir, f'{case_name}.nml')

    print('--- Step 1/3: Downloading ERA5 data ---')
    download_era5_time_series(
        start_date=start_date,
        end_date=end_date,
        lat=lat,
        lon=lon,
        output_dir=output_dir,
        name=case_name,
        source=source,
        stencil_step=stencil_step,
        max_workers=max_workers,
    )

    print('--- Step 2/3: Converting ERA5 to SCM forcings ---')
    era5_processed = _core_convert_forcings(era5_sfc_file, era5_pl_file)

    print('--- Step 3/3: Writing DEPHY output ---')
    _core_convert_to_dephy(
        era5_processed_forcings=era5_processed,
        start_date=start_date,
        case_name=case_name,
        output_file=dephy_file,
        template=template,
        column_area=column_area,
        sfc_roughness_length_cm=sfc_roughness_length_cm,
        namelist_file=nml_file,
        scm_cases_dir=scm_cases_dir,
        scm_config_dir=scm_config_dir,
    )
    print('--- Done ---')


if __name__ == "__main__":
    cli()
