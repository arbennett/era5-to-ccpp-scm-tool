#!/bin/bash

# Script to set up the Walnut Gulch FluxNet site case in the SCM

# Check if the case input data file exists
if [ ! -f "data/fluxnet_US-Whs.nc" ]; then
    echo "Error: Case input data file not found. Please run download_era5.sh and convert_forcing.sh first."
    exit 1
fi

# Copy the case configuration file to the etc/case_config/ directory
echo "Copying case configuration file to etc/case_config/ directory..."
cp fluxnet_US-Whs.nml ../../etc/case_config/

# Copy the run configuration file to the run/ directory
echo "Copying run configuration file to run/ directory..."
if [ ! -d "../../run" ]; then
    echo "Creating run directory..."
    mkdir -p ../../run
fi
cp fluxnet_US-Whs.ini ../../run/

# Check if the case input data file has been copied to the processed_case_input directory
if [ ! -f "../data/processed_case_input/fluxnet_US-Whs.nc" ]; then
    echo "Copying case input data file to data/processed_case_input/ directory..."
    cp data/fluxnet_US-Whs.nc ../../data/processed_case_input/
fi

echo "Setup complete. You can now run the SCM with the new case using:"
echo "python ../../bin/run_scm.py -c fluxnet_US-Whs -s SCM_GFS_v16"
