# Command line tool for converting ERA5 data to CCPP-SCM input

Builds ready-to-run [CCPP-SCM](https://ccpp-scm.readthedocs.io/en/latest/)
cases from ERA5 reanalysis for any point on the globe. Output is written in
DEPHY format and validated against CCPP-SCM v7.0.0.

Single-column models need large-scale forcing — advective tendencies, vertical
motion, geostrophic winds — that has traditionally been assembled by hand for
each study, which is why community case libraries hold only a few dozen curated
field-campaign cases. This tool generates a case for an arbitrary location and
period in one command, reading ERA5 from a public archive that needs no account
or API key.

## Installation
For now this tool can only be installed from source. To install it, clone the repository and run the following command in the root directory of the repository:

```bash
pip install .
```

For a development install, including the test dependencies:

```bash
uv sync --extra test
```

## Usage
Once installed the `era5-scm-tool` command will be available in your path. The tool has several subcommands. To see the available subcommands run:

```bash
era5-scm-tool --help
```

### Downloading ERA5 data

ERA5 is read from the [NSF NCAR ERA5 archive](https://registry.opendata.aws/nsf-ncar-era5/)
(RDA ds633.0), which is public: **no CDS account, API key, or request queue**.
The same archive is published in two places with an identical layout, selected
with `--source`:

| `--source` | Reads from | Notes |
|---|---|---|
| `aws` | `s3://nsf-ncar-era5` | Anonymous S3, works anywhere |
| `glade` | `/glade/campaign/collections/rda/data/d633000` | NCAR only, ~5x faster |
| `auto` (default) | Glade if visible, else S3 | |

```bash
era5-scm-tool download_era5 \
  --lat 31.7438 \
  --lon -110.0522 \
  --start_date 2019-01-01 \
  --end_date 2019-01-02 \
  --output_dir . \
  --name US-Whs \
  --source aws
```

This writes `US-Whs_pl.nc` (pressure-level), `US-Whs_sfc.nc` (surface,
radiative fluxes, and the soil, snow and albedo fields used to initialise the
land surface) and `US-Whs_inv.nc` (time-invariant land cover and soil texture).
The first two hold the 3x3 grid stencil centred on the site that the forcing
calculation needs for horizontal gradients.

#### A note on cost, and why you should batch sites

ERA5 in this archive is chunked one *whole global field* per time step, so
extracting a 3x3 stencil costs the same as reading the entire file — about
1.3 GB per pressure-level variable per day. That cost is per file, not per
site, so pulling many sites in one pass is much cheaper than looping. Use
`download_era5_sites` from Python for that:

```python
from era5_to_ccpp_scm.download_era5 import download_era5_sites

download_era5_sites(
    {"US-Whs": (31.7438, -110.0522),
     "US-MMS": (39.3232,  -86.4131),
     "US-Ha1": (42.5378,  -72.1715)},
    start_date="2019-06-01",
    end_date="2019-06-02",
    output_dir="./cases",
    source="aws",
)
```

Long time ranges are expensive from S3 for this reason; on NCAR machines
prefer `--source glade`.

### Converting ERA5 data to forcing data
To convert downloaded ERA5 surface + pressure-level files into processed SCM forcing fields:

```bash
era5-scm-tool convert_forcings --help
```

Example usage:

```bash
era5-scm-tool convert_forcings -s era5_sfc.nc -p era5_pl.nc -i era5_inv.nc \
  -o ccpp_scm_forcing.nc
```

### Writing the SCM case (DEPHY format)
`convert_to_dephy` writes the `*_SCM_driver.nc` file the SCM reads, plus its
companion case-config namelist, and can install both directly into an SCM
checkout:

```bash
era5-scm-tool convert_to_dephy \
  --era5_processed_forcings US-Whs_scm_forcings.nc \
  --start_date 2019-01-01 \
  --case_name fluxnet_US-Whs \
  --output_file fluxnet_US-Whs_SCM_driver.nc \
  --scm_cases_dir  "$SCM_ROOT/scm/data/processed_case_input" \
  --scm_config_dir "$SCM_ROOT/scm/etc/case_config"
```

`convert_era5_from_template` writes the older grouped format (`forcing`,
`initial`, `scalars` groups) and is kept only for backward compatibility;
current SCM releases read DEPHY.

### Nudging fields on different schedules

`--nudging_timescale_s` and `--nudging_above_pa` take a bare number, which
applies to every nudged field, or a per-field list:

```bash
era5-scm-tool convert_to_dephy \
  --era5_processed_forcings US-Jo2_scm_forcings.nc \
  --start_date 2020-08-01 \
  --case_name era5_optz_US-Jo2_exp2 \
  --output_file era5_optz_US-Jo2_exp2_SCM_driver.nc \
  --nudging_timescale_s 'ua=21600,va=21600,ta=21600,qt=43200' \
  --nudging_above_pa 85000
```

Fields are `ua`, `va`, `ta` and `qt` (`qv` is accepted for `qt`; CCPP-SCM v7.0.0
treats the two attributes identically). A field set to `off` is left free
running, and one left out of the list inherits the profile's timescale.

This is what lets a study isolate one field's constraint. Holding wind and
temperature on a fixed timescale while varying only the moisture relaxation
means a difference between two runs is attributable to the moisture treatment,
rather than to the whole column being anchored more or less tightly. From
Python, pass a mapping:

```python
convert_to_dephy(
    era5_processed=forcings,
    start_date="2020-08-01",
    case_name="era5_optz_US-Jo2_exp2",
    output_file="era5_optz_US-Jo2_exp2_SCM_driver.nc",
    nudging_timescale_s={"ua": 21600, "va": 21600, "ta": 21600, "qt": 43200},
    nudging_above_pa=85000.0,
)
```

### Running the full pipeline
Download, forcing conversion and DEPHY output in one command:

```bash
era5-scm-tool run_full_pipeline \
  --start_date 2019-01-01 \
  --end_date 2019-01-02 \
  --lat 31.7438 \
  --lon -110.0522 \
  --output_dir . \
  --case_name fluxnet_US-Whs \
  --source aws \
  --template gabls3
```

Then run it:

```bash
cd $SCM_ROOT/scm/bin
./run_scm.py -c fluxnet_US-Whs -s SCM_GFS_v16
```

## How the forcings are derived

- geostrophic winds from geopotential height gradients across the 3x3 stencil
- omega converted to vertical velocity (`w_ls`) with MetPy thermodynamic relations
- horizontal and vertical advective tendencies of `thetal` and `qt` computed on
  the stencil with spherical metric factors
- `dT_dt_rad` diagnosed as a column-mean heating rate from the TOA and surface
  net flux difference

## How the land surface is initialised

Soil, snow, albedo and land-cover state are taken from ERA5 at the site. ERA5
is produced with the HTESSEL land surface model, so its state has to be
reconciled with Noah before the SCM can use it:

- **Soil layers.** HTESSEL's interfaces are at 7, 28, 100 and 289 cm against
  Noah's 10, 40, 100 and 200 cm, so profiles are remapped by overlap weighting,
  which conserves the depth integral.
- **Soil moisture.** The two models assign different porosities and wilting
  points to the same soil. The default transfer preserves the degree of
  saturation between wilting point and porosity rather than the volumetric
  water content, because that ratio is what Noah's water stress factor is
  computed from. Use `--soil_moisture_transfer direct` for a clamped raw copy.
- **Frozen soil.** ERA5 archives total soil water; the liquid fraction is
  recovered from the freezing-point depression implied by the receiving soil's
  Clapp-Hornberger curve.
- **Classification.** HTESSEL land cover and soil texture are mapped onto the
  IGBP and STATSGO classes that the GFS physics reads (`ivegsrc = 1`,
  `isot = 1`). Green vegetation fraction is the ERA5 cover fraction closed by
  leaf area index, which supplies the seasonal cycle the cover fractions lack.

Skin temperature, deep soil temperature, snow water equivalent and depth, snow
cover, canopy water, the four albedo components and the roughness length are
taken from ERA5 directly. Fields ERA5 has no counterpart for (maximum snow
albedo, the `facsf`/`facwf` weights, friction velocity) still come from the
template. Every field records where it came from in the `land_state_source`
global attribute of the driver file.

### Overriding the site descriptors

ERA5 land cover is a 0.25 degree field, which is coarse next to a flux-tower
footprint. Where a site publishes its own classification, use it. Vegetation
type accepts an IGBP number, an IGBP class name, or a FLUXNET/AmeriFlux code:

```bash
# when building the case
era5-scm-tool run_full_pipeline ... --vegtyp OSH --soiltyp 'sandy loam'

# or afterwards, without repeating the download
era5-scm-tool set_land_state -f fluxnet_US-Whs_SCM_driver.nc \
    --vegtyp OSH --soiltyp 'sandy loam'
```

`set_land_state` also takes `--vegfrac`, `--shdmin`, `--shdmax`, `--slopetyp`,
`--zorl`, `--zorll`, `--snoalb`, `--tg3`, `--canopy`, `--facsf` and `--facwf`,
writes in place unless given `-o`, and records the change in both
`land_state_source` and the DEPHY `modifications` attribute.

Pass `--no_land` to skip the land download entirely and fall back to the
template, which reproduces the behaviour of releases before this feature.

## Run length, nudging, and the advection taper

A single-column model with prescribed forcing has nothing tying it to the
reanalysis it was built from, so errors accumulate. Over a day or two that does
not matter. Over weeks the column drifts and can destabilise. `--nudging`
chooses how much of it to hold:

| `--nudging` | what it does | when to use it |
|---|---|---|
| `none` (default) | nothing constrains the column | cases of a day or two |
| `free-troposphere` | relaxes above 700 hPa, boundary layer free | coupling work at sites with gentle terrain |
| `full-column` | relaxes everything | runs of a week or more, and any site with pronounced relief |

`free-troposphere` is the better choice on physical grounds, because the layer
the surface actually communicates with still evolves on its own. `full-column`
pulls near-surface temperature and humidity toward the reanalysis, so they no
longer respond freely to the land surface; the coupling signal is damped but
not removed, since surface fluxes are still diagnosed from the land state and
the soil still evolves on its own.

Override the timescale with `--nudging_timescale_s` (default 10800 s, three
hours) and the cutoff with `--nudging_above_pa`.

### The advection taper

`--advection_taper_pa` fades the advective tendencies and the pressure velocity
to zero over a given depth above the surface. Two reasons to use it:

- Horizontal gradients taken on a pressure surface close to the ground describe
  terrain rather than advection, because the surface cuts the hillsides
  differently at each stencil point. Over relief the spurious tendency does not
  average out.
- The pressure velocity has to vanish at the ground, and ERA5 does not deliver
  that where the lowest levels sit within the relief.

15000 Pa is a reasonable value. Measured at Walnut Gulch, where the surface
varies by 155 m across the 3x3 stencil, against Morgan Monroe at 37 m:

| | untapered near-surface tendency | monthly-mean pressure velocity |
|---|---|---|
| US-Whs | −2.7 to −5.2 K/day | −0.04 to −0.09 Pa/s |
| US-MMS | +0.7 to +1.3 K/day | ±0.018 Pa/s |

**The taper is not a substitute for nudging at a rough site.** At Walnut Gulch
the low-level advection is both terrain-contaminated and the real driver of the
January warming, so tapering it removes the signal along with the error and the
column cools. Measured against ERA5 2 m temperature over January:

| US-Whs configuration | bias | correlation |
|---|---|---|
| `free-troposphere` | −15.5 K | 0.04 |
| `free-troposphere` + taper | −11.9 K | 0.15 |
| `full-column` | −1.1 K | 0.95 |
| `full-column` + taper | **−0.6 K** | **0.95** |

So use the taper alongside nudging, not instead of it. At a gentle site like
Morgan Monroe every configuration works and the taper is close to neutral.

`paper/make_diagnostic_figure.py` plots near-surface air temperature against
ERA5 and prints these statistics, which is the check to run before trusting a
long case.

### Known limitations

- Land cover and soil texture come from ERA5's 0.25 degree grid unless
  overridden, which is coarse relative to a flux-tower footprint.
- Soil moisture transferred between the HTESSEL and Noah parameter tables is
  better read as an equivalent wetness than as a measured water content.
- Noah starts from a surface equilibrated by a different land surface model,
  which at some sites leaves a transient in the surface fluxes over the first
  model step. Discard it; lengthening the spinup does not remove it.
- `dT_dt_rad` is a bulk column-mean value; ERA5's archived fluxes are boundary
  values, so the vertical structure of radiative heating cannot be recovered
  from them. DEPHY cases set `radiation = "on"`, so the SCM computes radiation
  internally and ignores this field.
- Geostrophic winds are undefined at the equator and unreliable near it.

## Testing

The test suite is network-free — it runs against a small ERA5 extraction
committed under `casegen_walnut_gulch/data`, so it works offline:

```bash
uv run pytest
```

## Contributing and support

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to report a bug, ask a question,
or submit a change. Issues and questions go to the
[issue tracker](https://github.com/arbennett/era5-to-ccpp-scm-tool/issues).

## License

MIT — see [LICENSE](LICENSE).

## Repository layout

| Path | Contents |
|---|---|
| `era5_to_ccpp_scm/` | The installed package |
| `tests/` | Test suite |
| `examples/` | Shell examples for each subcommand |
| `scripts/` | Batch driver for many AmeriFlux sites |
| `contrib/derecho/` | Unsupported user scripts for NSF NCAR Derecho |
| `casegen_walnut_gulch/` | A worked example case, with test data |
| `paper/` | JOSS manuscript |

---

## Appendix: the legacy grouped input format

Everything below documents the **older** grouped NetCDF layout written by
`convert_era5_from_template`. Current SCM releases read DEPHY, so this is kept
only for backward compatibility — use `convert_to_dephy` for new work.

### Overall structure
The legacy CCPP-SCM input data is organized in a single netcdf file with multiple groups.
The group names are `forcing`, `initial`, and `scalars`, with a root group that I call `index`.

### Index data

- `soil_depth ('nsoil', )`: Depth of bottom of soil layers (m)

### Forcing data

- `p_surf ('time',)` :  surface pressure ( Pa )
- `T_surf ('time',)` :  surface absolute temperature ( K )
- `w_ls ('levels', 'time')` :  large scale vertical velocity ( m s^-1 )
- `omega ('levels', 'time')` :  large scale pressure vertical velocity ( Pa s^-1 )
- `u_g ('levels', 'time')` :  large scale geostrophic E-W wind ( m s^-1 )
- `v_g ('levels', 'time')` :  large scale geostrophic N-S wind ( - )
- `u_nudge ('levels', 'time')` :  E-W wind to nudge toward ( m s^-1 )
- `v_nudge ('levels', 'time')` :  N-S wind to nudge toward ( m s^-1 )
- `T_nudge ('levels', 'time')` :  absolute temperature to nudge toward ( K )
- `thil_nudge ('levels', 'time')` :  potential temperature to nudge toward ( K )
- `qt_nudge ('levels', 'time')` :  q_t to nudge toward ( kg kg^-1 )
- `dT_dt_rad ('levels', 'time')` :  prescribed radiative heating rate ( K s^-1, zero-filled if radiative inputs are missing )
- `h_advec_thetail ('levels', 'time')` :  prescribed theta_il tendency due to horizontal advection ( K s^-1 )
- `v_advec_thetail ('levels', 'time')` :  prescribed theta_il tendency due to vertical advection ( K s^-1 )
- `h_advec_qt ('levels', 'time')` :  prescribed q_t tendency due to horizontal advection ( kg kg^-1 s^-1 )
- `v_advec_qt ('levels', 'time')` :  prescribe q_t tendency due to vertical advection ( kg kg^-1 s^-1 )

### Initial condition data

- ` height ('levels',) `:  physical height at pressure levels ( m )
- ` thetail ('levels',) `:  initial profile of ice-liquid water potential temperature ( K )
- ` qt ('levels',) `:  initial profile of total water specific humidity ( kg kg^-1 )
- ` ql ('levels',) `:  initial profile of liquid water specific humidity ( kg kg^-1 )
- ` qi ('levels',) `:  initial profile of ice water specific humidity ( kg kg^-1 )
- ` u ('levels',) `:  initial profile of E-W horizontal wind ( m s^-1 )
- ` v ('levels',) `:  initial profile of N-S horizontal wind ( m s^-1 )
- ` tke ('levels',) `:  initial profile of turbulence kinetic energy ( m^2 s^-2 )
- ` ozone ('levels',) `:  initial profile of ozone mass mixing ratio ( kg kg^-1 )
- ` stc ('nsoil',) `:  initial profile of soil temperature ( K )
- ` smc ('nsoil',) `:  initial profile of soil moisture ( m3 m-3 )
- ` slc ('nsoil',) `:  initial profile of soil liquid moisture ( m3 m-3 )

### Scalars 

- ` alvsf () `:  60 degree vis albedo with strong cosz dependency ( - )
- ` alnsf () `:  60 degree nir albedo with strong cosz dependency ( - )
- ` alvwf () `:  60 degree vis albedo with weak cosz dependency ( - )
- ` alnwf () `:  60 degree nir albedo with weak cosz dependency ( - )
- ` facsf () `:  fractional coverage with strong cosz dependency ( - )
- ` facwf () `:  fractional coverage with weak cosz dependency ( - )
- ` vegfrac () `:  vegetation fraction ( - )
- ` canopy () `:  amount of water stored in canopy ( kg m-2 )
- ` f10m () `:  ratio of sigma level 1 wind and 10m wind ( - )
- ` t2m () `:  2-meter absolute temperature ( K )
- ` q2m () `:  2-meter specific humidity ( kg kg-1 )
- ` vegtyp () `:  vegetation type (1-12) ( - )
- ` soiltyp () `:  soil type (1-12) ( - )
- ` uustar () `:  friction velocity ( m s-1 )
- ` ffmm () `:  Monin-Obukhov similarity function for momentum ( - )
- ` ffhh () `:  Monin-Obukhov similarity function for heat ( - )
- ` hice () `:  sea ice thickness ( m )
- ` fice () `:  ice fraction ( - )
- ` tisfc () `:  ice surface temperature ( K )
- ` tprcp () `:  instantaneous total precipitation amount ( m )
- ` srflag () `:  snow/rain flag for precipitation ( - )
- ` snwdph () `:  water equivalent snow depth ( mm )
- ` shdmin () `:  minimum vegetation fraction ( - )
- ` shdmax () `:  maximum vegetation fraction ( - )
- ` slopetyp () `:  slope type (1-9) ( - )
- ` snoalb () `:  maximum snow albedo ( - )
- ` sncovr () `:  snow area fraction ( - )
- ` tsfcl () `:  surface skin temperature over land ( K )
- ` zorll () `:  surface roughness length over land ( cm )
- ` zorli () `:  surface roughness length over ice ( cm )
