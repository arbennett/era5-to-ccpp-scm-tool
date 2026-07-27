#!/bin/bash

# Script to download ERA5 data for the Walnut Gulch FluxNet site (US-Whs)

# Site coordinates
LATITUDE=31.7438
LONGITUDE=-110.0522

# Time period (adjust as needed)
START_DATE="2019-01-01"
END_DATE="2019-01-02"  # One month of data

# Output files
NAME="walnut_gulch"

# Create output directory if it doesn't exist
mkdir -p data

echo "Downloading ERA5 data for Walnut Gulch..."
era5-scm-tool download_era5 \
    --lat $LATITUDE \
    --lon $LONGITUDE \
    --start_date $START_DATE \
    --end_date $END_DATE \
    --output_dir data/ \
    --name $NAME 

echo "ERA5 data download complete."
