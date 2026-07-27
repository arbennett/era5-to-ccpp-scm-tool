#!/usr/bin/env bash
# Write the DEPHY driver plus its case-config namelist, and install both
# straight into an SCM checkout.
set -euo pipefail

SCM_ROOT=${SCM_ROOT:-/path/to/ccpp-scm}

era5-scm-tool convert_to_dephy \
  --era5_processed_forcings US-Whs_scm_forcings.nc \
  --start_date 2019-01-01 \
  --case_name fluxnet_US-Whs \
  --output_file fluxnet_US-Whs_SCM_driver.nc \
  --scm_cases_dir  "${SCM_ROOT}/scm/data/processed_case_input" \
  --scm_config_dir "${SCM_ROOT}/scm/etc/case_config"
