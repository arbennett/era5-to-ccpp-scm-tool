#!/usr/bin/env bash
# Download ERA5, compute forcings and write a DEPHY case the SCM can run.
set -euo pipefail

era5-scm-tool run_full_pipeline \
  --start_date 2019-01-01 \
  --end_date 2019-01-02 \
  --lat 31.7438 \
  --lon -110.0522 \
  --output_dir . \
  --case_name fluxnet_US-Whs \
  --source aws \
  --template gabls3
