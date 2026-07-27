#!/bin/bash
set -e

# Script to convert ERA5 data to SCM forcing format for the Walnut Gulch FluxNet site (US-Whs)

# Input files
NAME="walnut_gulch"
INPUT_SFC="data/${NAME}_sfc.nc"
INPUT_RAD="data/${NAME}_rad.nc"
INPUT_PL="data/${NAME}_pl.nc"

# Output files
OUTPUT_FORCINGS="data/${NAME}_scm_forcings.nc"
OUTPUT_SCM="data/fluxnet_US-Whs.nc"

# Check if input files exist
if [ ! -f "$INPUT_SFC" ] || [ ! -f "$INPUT_PL" ]; then
    echo "Error: Input files not found. Please run download_era5.sh first."
    exit 1
fi

echo "Converting ERA5 data to SCM forcing format..."
era5-scm-tool convert_forcings \
    --era5_surface_file $INPUT_SFC \
    --era5_rad_file $INPUT_RAD \
    --era5_pressure_levels_file $INPUT_PL \
    --output_file $OUTPUT_FORCINGS

echo "Forcings conversion complete."
echo "SCM forcings file: $OUTPUT_FORCINGS"

echo "Creating SCM input file from template..."
era5-scm-tool convert_era5_from_template \
    --era5_processed_forcings $OUTPUT_FORCINGS \
    --output_file $OUTPUT_SCM \
    --template gabls3

echo "SCM input file creation complete."
echo "SCM input file: $OUTPUT_SCM"

# Copy the output file to the SCM processed_case_input directory
echo "Copying SCM input file to processed_case_input directory..."
cp $OUTPUT_SCM ../casegen_walnut_gulch/data/

echo "Setup complete. You can now run the SCM with the new case."
echo "Don't forget to copy the case configuration file to etc/case_config/ and the run configuration file to run/."
