import click
import xarray as xr
import numpy as np

from . import templates
from .util import _maybe_open
from typing import Union, Optional
from .download_era5 import download_era5_time_series
from .convert_forcing import era5_to_scm_forcing

@click.group()
def cli():
    """
    ERA5 to CCPP-SCM data processing CLI. Commands are:
     1. download_era5
     2. generate_template
     3. convert_forcings
     4. all
    """
    pass


@cli.command(name='download_era5')
@click.option('--start_date', type=str)
@click.option('--end_date', type=str)
@click.option('--lat', type=float)
@click.option('--lon', type=float)
@click.option('--output_dir', type=str)
@click.option('--name', type=str, default='era5')
def download_era5(
    start_date: str,
    end_date: str,
    lat: float,
    lon: float,
    output_dir: str,
    name: str = 'era5'
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
        name=name
    )

def _core_convert_forcings(
    era5_surface_file: Union[str, xr.Dataset],
    era5_pressure_levels_file: Union[str, xr.Dataset],
    output_file: Optional[str]=None
):
    ds1 = _maybe_open(era5_surface_file)
    ds2 = _maybe_open(era5_pressure_levels_file)
    ds = xr.merge([ds1, ds2])
    rename_map = {}
    if "valid_time" in ds.dims or "valid_time" in ds.coords:
        rename_map["valid_time"] = "time"
    if "pressure_level" in ds.dims or "pressure_level" in ds.coords:
        rename_map["pressure_level"] = "levels"
    if rename_map:
        ds = ds.rename(rename_map)
    out = era5_to_scm_forcing(ds)
    if output_file is not None:
        out.to_netcdf(output_file, format="NETCDF4")
    return out


@cli.command(name='convert_forcings')
@click.option('-s', '--era5_surface_file', type=str)
@click.option('-p', '--era5_pressure_levels_file', type=str)
@click.option('-o', '--output_file', type=str)
def convert_forcings(
    era5_surface_file: Union[str, xr.Dataset],
    era5_pressure_levels_file: Union[str, xr.Dataset],
    output_file: Optional[str]=None,
):
    """
    Convert ERA5 data to CCPP-SCM forcing file.
    """
    return _core_convert_forcings(era5_surface_file, era5_pressure_levels_file, output_file)


def _core_convert_era5_from_template(
    era5_processed_forcings: Union[str, xr.Dataset],
    template: str,
    output_file: str
):
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

    # Write forcing variables by matching template var shapes/dims.
    forcing_out = template_forcing.copy(deep=True)
    for var in template_forcing.data_vars:
        if var not in era5_ds:
            continue
        da = era5_ds[var]
        if set(template_forcing[var].dims).issubset(set(da.dims)):
            da = da.transpose(*template_forcing[var].dims)
            forcing_out[var] = xr.DataArray(
                da.values.astype(template_forcing[var].dtype, copy=False),
                dims=template_forcing[var].dims,
                coords={d: forcing_out.coords[d] for d in template_forcing[var].dims if d in forcing_out.coords},
                attrs=template_forcing[var].attrs,
            )

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
    Generate a template for CCPP-SCM input file.
    """
    _core_convert_era5_from_template(era5_processed_forcings, template, output_file)

@cli.command(name='run_full_pipeline')
@click.option('--start_date', type=str)
@click.option('--end_date', type=str)
@click.option('--lat', type=float)
@click.option('--lon', type=float)
@click.option('--output_dir', type=str, default='.')
@click.option('--name', type=str, default='era5')
@click.option('--template', type=str, default='gabls3')
def run_full_pipeline(
    start_date: str,
    end_date: str,
    lat: float,
    lon: float,
    output_dir: str,
    name: str = 'era5',
    template: str = 'gabls3'
):
    """
    Run the full pipeline from downloading ERA5 data to generating a CCPP-SCM input file.
    """
    print('Downloading ERA5 data...')
    download_era5_time_series(
        start_date=start_date,
        end_date=end_date,
        lat=lat,
        lon=lon,
        output_dir=output_dir,
        name=name
    )
    print('Downloaded ERA5 data')
    print('-------------------------------------')
    output_file = f'{output_dir}/{name}_scm.nc'
    era5_pl_file = f'{output_dir}/{name}_pl.nc'
    era5_sfc_file = f'{output_dir}/{name}_sfc.nc'
    print(era5_pl_file)
    print(era5_sfc_file)
    print('converting forcings...')
    era5_processed = _core_convert_forcings(era5_sfc_file, era5_pl_file)
    print('converted forcings')
    print('-------------------------------------')
    print('converting to template...')
    _core_convert_era5_from_template(era5_processed, template, output_file)
    print('converted to template')
    print('-------------------------------------')




if __name__ == "__main__":
    cli()
