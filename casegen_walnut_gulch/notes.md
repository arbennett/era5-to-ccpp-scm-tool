# Understanding the SCM Codebase and Implementing a Fluxnet Site Case

First, the differences between a "run", "case", "suite", and "experiment" in this codebase:

Case: A case represents a specific atmospheric scenario with its own configuration and input data. It includes:

- Initial conditions (temperature, humidity, wind profiles, etc.)
- Forcing data (large-scale vertical velocity, advection, etc.)
- Surface conditions
- Case-specific parameters (location, time period, etc.)

Cases are defined by:

- A case configuration file in `etc/case_config/` (e.g., `gabls3.nml`)
- Input data files in `data/processed_case_input/` (e.g., `gabls3.nc`)

Suite: A suite is a collection of physics parameterization schemes organized in a specific way. It defines:

- Which physics schemes are used (radiation, PBL, convection, microphysics, etc.)
- The order in which they are called
- How they interact with each other

Suites are defined in XML files like `suite_SCM_GFS_v16.xml`.

Run: A run is a single execution of the model with a specific case and suite. It can be configured with additional options like:

- Runtime duration
- Output frequency
- Vertical levels
- Timestep

Experiment: An experiment is essentially a combination of a case and a suite with specific configuration options. In the code, the Experiment class in run_scm.py represents this concept. It handles:

- Setting up the run directory
- Linking necessary files
- Creating output directories
- Configuring the model for execution

# Implementing a New Fluxnet Site Case
To implement a new case using fluxnet sites with your ERA5 extraction tool, you'll need to follow these steps:

## 1. Extract ERA5 Data for Fluxnet Sites
I built the `era5-to-ccpp-scm-tool` for this task. For more information on the variables used to run the SCM see the section on "Variables Required for the GABLS3_NOAHMP Case". The `lera5-to-ccpp-scm-tool` is designed to:

 - Download ERA5 data for a specific location and time period
 - Convert it to the format required by the SCM

## 2. Create a New Case Configuration File
You'll need to create a new case configuration file in `etc/case_config/`. For example, if you're creating a case for a fluxnet site called "US-Whs", you might create `etc/case_config/fluxnet_US-Whs.nml`.

Based on the example of `gabls3.nml`, this should look something like:

```
$case_config
case_name = 'fluxnet_US-Whs',
runtime = 86400,  # Adjust as needed (in seconds)
thermo_forcing_type = 2,
mom_forcing_type = 2,
relax_time = 7200.0,
sfc_flux_spec = .false.,
lsm_ics = .true.,
do_spinup = .true.,
spinup_timesteps = 12,
sfc_roughness_length_cm = 15.0,  # Adjust based on site characteristics
sfc_type = 1,
reference_profile_choice = 2,
year = 2019,  # Adjust based on your data
month = 1,    # Adjust based on your data
day = 1,      # Adjust based on your data
hour = 0,     # Adjust based on your data
column_area = 1.45E8,
$end
```
You'll need to adjust the parameters based on the specific fluxnet site characteristics.

## 3. Place the Processed Case Input Data
After converting the ERA5 data to SCM format, you'll need to place the resulting netCDF file in the `data/processed_case_input/` directory. The filename should match the case name, e.g., `fluxnet_US-Whs.nc`.

The structure of this file is critical and must include:

- Forcing data (large-scale vertical velocity, advection, etc.)
- Initial conditions (temperature, humidity, wind profiles, etc.)
- Surface conditions

The era5-to-ccpp-scm-tool should handle creating this file in the correct format.

## 4. Create a Run Configuration
You can create a run configuration file in the run/ directory, e.g., `fluxnet_US-Whs.ini`. This file configures how the output is processed and visualized.

## 5. Run the Model with Your New Case
Once everything is set up, you can run the model with your new case using:

```bash
python bin/run_scm.py -c fluxnet_US-Whs -s SCM_GFS_v16
```

This will:

- Set up the run directory with all necessary files
- Execute the model with your case and the specified suite
- Generate output in the specified output directory


# Variables Required for the `GABLS3_NOAHMP` Case
Based on the documentation in the SCM Technical Guide, these are the variables needed for the `gabls3_noahmp` case.

The case input data file for the SCM is organized into 3 main groups:

## 1. Index Data
soil_depth: Depth of bottom of soil layers (m)

## 2. Initial Condition Data
- height: Physical height at pressure levels (m)
- thetail: Initial profile of ice-liquid water potential temperature (K)
- qt: Initial profile of total water specific humidity (kg kg^-1)
- ql: Initial profile of liquid water specific humidity (kg kg^-1)
- qi: Initial profile of ice water specific humidity (kg kg^-1)
- u: Initial profile of E-W horizontal wind (m s^-1)
- v: Initial profile of N-S horizontal wind (m s^-1)
- tke: Initial profile of turbulence kinetic energy (m^2 s^-2)
- ozone: Initial profile of ozone mass mixing ratio (kg kg^-1)
- stc: Initial profile of soil temperature (K)
- smc: Initial profile of soil moisture (m3 m-3)
- slc: Initial profile of soil liquid moisture (m3 m-3)

## 3. Forcing Data
- p_surf: Surface pressure (Pa)
- T_surf: Surface absolute temperature (K)
- w_ls: Large scale vertical velocity (m s^-1)
- omega: Large scale pressure vertical velocity (Pa s^-1)
- u_g: Large scale geostrophic E-W wind (m s^-1)
- v_g: Large scale geostrophic N-S wind (-)
- u_nudge: E-W wind to nudge toward (m s^-1)
- v_nudge: N-S wind to nudge toward (m s^-1)
- T_nudge: Absolute temperature to nudge toward (K)
- thil_nudge: Potential temperature to nudge toward (K)
- qt_nudge: q_t to nudge toward (kg kg^-1)
- dT_dt_rad: Prescribed radiative heating rate (K s^-1)
- h_advec_thetail: Prescribed theta_il tendency due to horizontal advection (K s^-1)
- v_advec_thetail: Prescribed theta_il tendency due to vertical advection (K s^-1)
- h_advec_qt: Prescribed q_t tendency due to horizontal advection (kg kg^-1 s^-1)
- v_advec_qt: Prescribed q_t tendency due to vertical advection (kg kg^-1 s^-1)

## 4. Scalar Data (especially important for NoahMP)
- alvsf: 60 degree vis albedo with strong cosz dependency (-)
- alnsf: 60 degree nir albedo with strong cosz dependency (-)
- alvwf: 60 degree vis albedo with weak cosz dependency (-)
- alnwf: 60 degree nir albedo with weak cosz dependency (-)
- facsf: Fractional coverage with strong cosz dependency (-)
- facwf: Fractional coverage with weak cosz dependency (-)
- vegfrac: Vegetation fraction (-)
- canopy: Amount of water stored in canopy (kg m-2)
- f10m: Ratio of sigma level 1 wind and 10m wind (-)
- t2m: 2-meter absolute temperature (K)
- q2m: 2-meter specific humidity (kg kg-1)
- vegtyp: Vegetation type (1-12) (-)
- soiltyp: Soil type (1-12) (-)
- uustar: Friction velocity (m s-1)
- ffmm: Monin-Obukhov similarity function for momentum (-)
- ffhh: Monin-Obukhov similarity function for heat (-)
- hice: Sea ice thickness (m)
- fice: Ice fraction (-)
- tisfc: Ice surface temperature (K)
- tprcp: Instantaneous total precipitation amount (m)
- srflag: Snow/rain flag for precipitation (-)
- snwdph: Water equivalent snow depth (mm)
- shdmin: Minimum vegetation fraction (-)
- shdmax: Maximum vegetation fraction (-)
- slopetyp: Slope type (1-9) (-)
- snoalb: Maximum snow albedo (-)
- sncovr: Snow area fraction (-)
- tsfcl: Surface skin temperature over land (K)
- zorll: Surface roughness length over land (cm)
- zorli: Surface roughness length over ice (cm)

### Important Notes:
 - Not all variables are required for every case. The specific variables needed depend on the forcing type specified in the case configuration file:
    - For thermo_forcing_type = 1: horizontal and vertical advective tendencies of θ_il and q_t and prescribed radiative heating
    - For thermo_forcing_type = 2: horizontal advective tendencies of θ_il and q_t, prescribed radiative heating, and the large scale vertical pressure velocity
    - For thermo_forcing_type = 3: θ_il and q_t nudging profiles and the large scale vertical pressure velocity
    - For mom_forcing_type = 2: geostrophic winds and large scale vertical velocity
    - For mom_forcing_type = 3: u and v nudging profiles
 - Since gabls3_noahmp uses the Noah-MP land surface model, it requires additional soil and surface variables compared to cases that don't use a land surface model.
 - The ERA5-to-CCPP-SCM tool you're using should handle the conversion of ERA5 data to the required format, including calculating derived variables like ice-liquid water potential temperature (θ_il) from the basic ERA5 variables.