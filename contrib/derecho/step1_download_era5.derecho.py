import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import cmocean
from nco import Nco
from nco.custom import Limit, LimitSingle
import calendar
import timeit
import os
import dask
from dask_jobqueue import PBSCluster
from dask.distributed import Client,LocalCluster
import sys
import glob

# How to run this script
# python download_era5.derecho.py <site>
# <site> is the AmeriFlux site ID, e.g. US-Ha1

#from dask_jobqueue import PBSCluster
cluster = PBSCluster(
    cores=15,
    memory="250GB",
    processes=15,
    queue="main",
    account="UCUB0117",
    walltime="12:00:00",
    job_extra_directives=["-l job_priority=premium"]
)
cluster.scale(jobs=3)
client = cluster.get_client()

print("total number of workers is %i"%(len(client.scheduler_info()["workers"])))

# environment for nco
os.environ["NCOpath"]='/glade/work/yifanc/anaconda3/envs/nco/bin'
# location for all locations
ameriflux_df = pd.read_csv("ameriflux.sites.north_hem.nl5yr.location.csv",index_col=0)
site = sys.argv[1] #'US-Ha1'
site_lat = ameriflux_df['Latitude (degrees)'].loc[site]
site_lon = ameriflux_df['Longitude (degrees)'].loc[site]
xcel_input = pd.ExcelFile("ERA5_variable.SCM.input.xlsx")
var_df = xcel_input.parse(sheet_name='era5_var')
delta=0.375 # 1.5 times the grid cells
iroot_dir = "/glade/campaign/collections/rda/data/d633000/"
oroot_raw_dir = "/glade/derecho/scratch/yifanc/wpo2023/SCM/raw/"
osite_raw_dir = oroot_raw_dir + "%s/"%site
if not os.path.exists(osite_raw_dir):
    os.mkdir(osite_raw_dir)
# create the time list
# month_list = pd.date_range("2019-01-01","2024-12-31",freq='1MS')
# print(month_list)
# for date in month_list:
#     year = date.year
#     month = date.month
#     ystr = str(year)
#     mstr = "%02i"%month
#     yms = "%s%s"%(ystr,mstr)
#     print(ystr, mstr, yms)

for year in range(2019,2025):
    for row in var_df.iterrows():
        variable = row[1]['Variable']
        prefix = row[1]['prefix']
        code_table = row[1]['code_table']
        param_code = "%03i"%row[1]['param_code']
        var_abbr = row[1]['var_abbr']
        postfix = row[1]['postfix']
        agg_time = row[1]['agg_time']
        print(variable, prefix, code_table, param_code, var_abbr, postfix)
        # file name
        infile_month = iroot_dir + "%s/%i*/%s.%i_%s_%s.%s.*.nc"%(prefix,year,prefix,code_table,param_code,var_abbr,postfix)
        ofile_month  = osite_raw_dir + "%s.%i_%s_%s.%s.%s.%i.nc"%(prefix,code_table,param_code,var_abbr,postfix,site,year)
        if not os.path.isfile(ofile_month):
            # read in data
            time1 = timeit.default_timer()
            infile_all = glob.glob(infile_month)
            if prefix == "e5.oper.an.pl":
                trunk_size=4
            elif agg_time == "time":
                trunk_size=120
            else:
                trunk_size=12
            ds_month = xr.open_mfdataset(infile_all,
                                         concat_dim=agg_time,
                                         combine="nested",
                                         parallel=True,
                                         engine="netcdf4",
                                         chunks={agg_time: trunk_size})
            ds_month_sel = ds_month.sel(latitude=slice(site_lat+delta,site_lat-delta),
                                        longitude=slice(site_lon%360-delta,site_lon%360+delta))
            ds_month_sel.to_netcdf(ofile_month)


