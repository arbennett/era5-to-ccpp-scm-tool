#!/usr/bin/env bash
# Download ERA5, compute forcings and write a DEPHY case the SCM can run.
set -euo pipefail

era5-scm-tool run_full_pipeline \
  --start_date 2019-01-01 \
  --end_date 2019-01-05 \
  --lat 31.7438 \
  --lon -110.0522 \
  --output_dir . \
  --case_name fluxnet_US-Whs \
  --source aws \
  --template gabls3

# The land surface is initialised from ERA5 by default. ERA5 land cover is a
# 0.25 degree field, so where the site publishes its own class, use it. This
# patches the finished case without repeating the download.
era5-scm-tool set_land_state \
  --driver_file fluxnet_US-Whs_SCM_driver.nc \
  --vegtyp OSH
