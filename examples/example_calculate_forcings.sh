#!/usr/bin/env bash
# Turn raw ERA5 surface + pressure-level files into SCM forcing fields.
# The invariant file is optional; without it the land cover and soil texture
# classes fall back to the case template.
set -euo pipefail

era5-scm-tool convert_forcings \
  --era5_surface_file US-Whs_sfc.nc \
  --era5_pressure_levels_file US-Whs_pl.nc \
  --era5_invariant_file US-Whs_inv.nc \
  --output_file US-Whs_scm_forcings.nc
