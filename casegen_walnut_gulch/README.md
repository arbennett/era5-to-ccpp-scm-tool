# Walnut Gulch FluxNet Site (US-Whs) Case Generation

This directory contains scripts and configuration files for generating a CCPP-SCM case for the Walnut Gulch FluxNet site (US-Whs) in southern Arizona.

## Site Information

- **FluxNet Code**: US-Whs
- **Latitude**: 31.7438° N
- **Longitude**: -110.0522° E
- **Elevation**: 1370m
- **Location**: Walnut Gulch, Arizona, USA

## Directory Contents

- `README.md`: This documentation file
- `download_era5.sh`: Script to download ERA5 data for the site
- `convert_forcing.sh`: Script to convert ERA5 data to SCM format
- `setup_case.sh`: Script to set up the case in the SCM
- `fluxnet_US-Whs.nml`: Case configuration file (to be placed in etc/case_config/)
- `fluxnet_US-Whs.ini`: Run configuration file (to be placed in run/)
- `data/`: Directory containing downloaded and converted data

## Case Generation Process

1. Download ERA5 data for the Walnut Gulch location using the era5-to-ccpp-scm-tool
2. Convert the ERA5 data to SCM forcing format
3. Create the SCM input file from a template
4. Create the case configuration file with appropriate parameters
5. Create the run configuration file for visualization
6. Place the processed case input data in the correct location
7. Test the case by running the SCM

## Usage

1. Run the download script to get ERA5 data:
   ```
   ./download_era5.sh
   ```

2. Run the conversion script to create SCM input:
   ```
   ./convert_forcing.sh
   ```

3. Run the setup script to set up the case in the SCM:
   ```
   ./setup_case.sh
   ```

4. Run the SCM with the new case:
   ```
   python ../bin/run_scm.py -c fluxnet_US-Whs -s SCM_GFS_v16
