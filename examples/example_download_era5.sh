#!/usr/bin/env bash
# Extract a 3x3 ERA5 stencil for one site from the public NSF NCAR archive.
# No CDS account or API key is needed.
set -euo pipefail

era5-scm-tool download_era5 \
  --lat 31.7438 \
  --lon -110.0522 \
  --start_date 2019-01-01 \
  --end_date 2019-01-02 \
  --output_dir . \
  --name US-Whs \
  --source aws
