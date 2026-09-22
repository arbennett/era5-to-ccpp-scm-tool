---
title: 'era5-to-ccpp-scm-tool: single-column model cases from ERA5 reanalysis for any location on Earth'
tags:
  - Python
  - atmospheric science
  - single-column model
  - reanalysis
  - ERA5
  - parameterization
authors:
  # PLACEHOLDER: confirm ORCIDs and affiliations before submission.
  - name: Andrew Bennett
    orcid: 0000-0000-0000-0000
    corresponding: true
    affiliation: 1
  - name: Nabin Kalauni
    orcid: 0000-0000-0000-0000
    corresponding: true
    affiliation: 1
  - name: Yifan Cheng
    orcid: 0000-0000-0000-0000
    affiliation: 2
  - name: Ronnie Abolafia-Rosenzweig
    orcid: 0000-0000-0000-0000
    affiliation: 3
  - name: Kathryn Newman
    orcid: 0000-0000-0000-0000
    affiliation: 3
  - name: Andrew Newman
    orcid: 0000-0000-0000-0000
    affiliation: 3
affiliations:
  - name: University of Arizona, Tucson, AZ, USA
    index: 1
  - name: University at Buffalo, Buffalo, NY, USA
    index: 2
  - name: National Center for Atmospheric Research, Boulder, CO, USA
    index: 3
date: 27 July 2026
bibliography: paper.bib
---

# Summary

Single-column models (SCMs) simulate the physics of an atmospheric model,
including radiation, turbulence, convection, cloud microphysics, and the land
surface, for a single vertical column, with the effects of the resolved
dynamics supplied as prescribed forcing rather than computed by the model
itself [@randall1996]. Because single column models are computationally frugal, they are widely used for developing and testing parameterizations,
and for examining what a physics suite does at a particular place and time.
Running an SCM requires a case, which consists of initial profiles of the
atmospheric state, large-scale advective tendencies of heat and moisture,
vertical motion, geostrophic winds, and surface and land-surface properties,
all assembled in the layout that the target model reads. Building such a case
has traditionally been an ad hoc, per-study effort, and as a result the case
libraries that the community shares consist of a few dozen curated field
campaign cases.

`era5-to-ccpp-scm-tool` is a Python package and command line tool that builds
SCM cases for the Common Community Physics Package
Single Column Model (CCPP-SCM) [@heinzeller2023; @ccppscm] from ERA5 reanalysis [@hersbach2020] for any location on the globe.
Given a latitude, longitude, and date range, it extracts the reanalysis fields
that are needed, derives the full set of large-scale forcing terms, and writes
a case in the DEPHY format [@dephy] along with its configuration namelist, so
that the case can be run directly in the CCPP-SCM. A complete case is
generated with a single command:

```bash
era5-scm-tool run_full_pipeline \
  --start_date 2019-01-01 --end_date 2019-01-02 \
  --lat 31.7438 --lon -110.0522 \
  --case_name fluxnet_US-Whs --output_dir .
```

We have designed the package to fit into the broader scientific Python
ecosystem, building on popular packages such as xarray [@hoyer2017], NumPy
[@harris2020], and MetPy [@may2022], and it provides both a command line tool
as well as API access as a Python module.

# Statement of need

The development of this tool was motivated by gaps in how the driving data are
obtained and in how the forcing terms are derived. ERA5 is normally accessed
through the Copernicus Climate Data Store, which requires an account, an API
key, and a request queue whose latency is neither bounded nor reproducible.
This is a poor fit for batch work and an awkward dependency for a published
workflow. We instead read the NSF NCAR ERA5 archive [@rda633], which is public,
requires no authentication, and is published in two places with an identical
directory layout: an anonymous S3 bucket that can be read from anywhere, and a
filesystem mount on NSF NCAR machines. We verified that the two backends
produce bit-identical extractions, so a workflow developed on a laptop
reproduces exactly on a high-performance computing system. Forcing derivations such as this are usually implemented as ad hoc solutions based on
similar sets of equations, and our goal was to formalize the process by developing this package.

We chose CCPP-SCM as the target model because of its lineage. CCPP-SCM is the
single-column driver for the Common Community Physics Package
[@heinzeller2023], and the physics suites that it runs are the same source
code, invoked through the same framework, as the physics in NOAA's operational
Unified Forecast System. Suites are selected at
run time from suite definition files, which makes it inexpensive to push a
single case through several physics configurations and to attribute the
differences to the physics rather than to the setup [@groot2026]. CCPP-SCM also couples a
full land surface model [Noah-MP, @niu2011], which is a requirement for the
land-atmosphere coupling problems that motivated this work, and it reads DEPHY
[@dephy], a community format, so the cases that this tool writes are not
restricted to a single model.

Existing tools for building single-column cases do not fill this gap. SCAM6
[@gettelman2019] is distributed with the CESM physics that it is built around,
and the DEPHY case library distributes curated field campaign cases such as
GABLS3 [@bosveld2014]. Both of these constrain a study to the locations and
periods that someone has already prepared, while the application that motivated
this work, evaluating land-atmosphere coupling against flux tower observations
[@novick2018; @pastorello2020], requires cases at arbitrary sites generated in
bulk. CCPP-SCM itself ships a case generator, `UFS_case_gen.py`, which builds
DEPHY cases from Unified Forecast System initial condition and history files on
the native cubed-sphere grid [@ccppscm]. That script requires the user to
already hold a matched set of initial conditions, supergrid files, and history
output, and it is the appropriate choice when the question concerns the
forecast system itself. It draws on a forecast archive, however, which is
produced by an operational configuration that changes with each upgrade, so a
record assembled across several years is not homogeneous. ERA5 is a reanalysis,
in which a single frozen model and assimilation system is applied across the
whole 1940 to present record while ingesting the observational record as it
goes [@hersbach2020]. For studies that span many sites and many years, that
observational constraint and temporal homogeneity are of interest, making the two tools complementary rather than competing.

# Methods

ERA5 and a DEPHY case file differ in many conventions, and the
conversions between them fall into six broad groups.

The first group concerns coordinates and indexing. ERA5 latitudes are stored
north first, so a northward gradient is computed as the difference between the
first and third rows of the stencil rather than the reverse. Longitudes run
from 0 to 360 degrees and are unwrapped when a site lies near the prime
meridian, and pressure levels arrive in hPa and are re-indexed onto their
converted values rather than simply rescaled.

The second group concerns time. The accumulated radiative fluxes are archived
as hourly totals in $J/m^{2}$ and are divided by the accumulation period to
give rates in $W/m^{2}$. Those fluxes come from a forecast stream dimensioned
by forecast initial time and forecast hour, which is flattened onto valid time
to give an hourly series. The absolute timestamps that ERA5 carries
are then converted to seconds since the start of the case, which is the
convention DEPHY uses.

The third group is thermodynamic. Temperature and specific humidity are
converted to liquid water potential temperature and total water specific
humidity, the pressure velocity that ERA5 archives is converted to a vertical
velocity in $m/s$ using the thermodynamic relations provided by MetPy, and
geopotential is converted to geopotential height.

The fourth group is dynamical. Geostrophic winds are computed from the
gradients of geopotential height across the stencil, and the horizontal and
vertical advective tendencies of liquid water potential temperature and total
water are computed using three point derivatives on a non-uniform stencil with
spherical metric factors applied. A column mean radiative heating rate is
diagnosed from the difference between the net radiative fluxes at the top of
the atmosphere and at the surface.

The fifth group concerns the land surface. ERA5 is produced with the HTESSEL
land surface model [@balsamo2009], whose state cannot be handed to Noah or
Noah-MP unchanged, and reconciling the two requires three separate
transformations. The four soil layers that HTESSEL carries have interfaces at
7, 28, 100 and 289 cm against the 10, 40, 100 and 200 cm that Noah uses, so
soil temperature and moisture are remapped by overlap weighting, which
conserves the depth integral of the remapped quantity. The two models also
assign different porosities and wilting points to the same soil, so carrying a
volumetric water content across unchanged can place it outside the physical
range of the receiving soil type, and we therefore transfer the degree of
saturation between the wilting point and the porosity of each parameter set
rather than the water content itself. This preserves the ratio from which Noah
computes its water stress factor, and with it the evaporative regime. The
liquid water fraction that Noah carries separately is recovered from the
freezing point depression implied by the Clapp-Hornberger retention curve of
the receiving soil. Land cover and soil texture are mapped from the HTESSEL
classes onto the IGBP and STATSGO classifications that the GFS physics is
configured to read, and the green vegetation fraction is obtained by closing
the ERA5 cover fractions with the leaf area index, which supplies the seasonal
cycle that the cover fractions themselves lack.

The final group concerns the schema. ERA5 short names are mapped onto DEPHY
variable names, and the grouped arrays that older CCPP-SCM releases used are
written into the flat layout that DEPHY specifies. This conversion is not
purely cosmetic, because DEPHY encodes which forcing terms are active in the
global attributes of the case file. The attributes are set to enable advection
of liquid water potential temperature and total water, to supply pressure
velocity and geostrophic forcing, to enable the internal radiation scheme, and
to declare whatever nudging the user has asked for. A case whose attributes disagree with the arrays it
carries will run to completion and produce a plausible but incorrect answer,
which is one of the reasons we consider it worthwhile to package these steps
together. The companion Fortran namelist that declares the case to the SCM is
written at the same stage.

## Configuring a case

A column driven by prescribed forcing has nothing tying it to the reanalysis it
was derived from, so the errors that any parameterization makes accumulate
without check. Over a day or two this is unimportant, and it is the
configuration in which most single column cases are used. Over weeks the column
drifts away from the reanalysis and can leave the range in which the physics is
numerically well behaved. This package therefore offers two controls,
both off by default so that a short case is unaffected.

The first relaxes the column toward the reanalysis profiles, which the DEPHY
format supports directly and which the package writes from the same fields it
already derives. Three settings are available. Relaxing nothing is right for a
case of a day or two. Relaxing only the free troposphere, above 700 hPa, leaves
the boundary layer to evolve on its own, which is what we would choose for
land-atmosphere work because it is the layer the surface communicates with.
Relaxing the whole column is the robust choice for longer runs, at the cost of
pulling near-surface temperature and humidity toward the reanalysis so that
they no longer respond freely to the land surface. That cost is real but
bounded: surface fluxes are still diagnosed from the land state, and the soil
still evolves on its own, so what the land surface does remains visible even
when the air above it is constrained.

The second control fades the advective tendencies and the pressure velocity to
zero over a chosen depth above the ground. Two things motivate it. A horizontal
gradient taken on a pressure surface lying close to the ground describes the
terrain rather than the flow, because the surface intersects the hillsides at a
different height at each point of the stencil, and over pronounced relief the
resulting spurious tendency does not average away. The pressure velocity must
in any case vanish at the ground, which ERA5 does not deliver where the lowest
levels sit within the relief. At US-Whs, where the surface varies by 155 m
across the stencil, the untapered fields carry a sustained 3 to 5 K per day of
low level cooling and a monthly mean pressure velocity near 0.09 Pa s $^{-1}$ of
ascent, against 1 K per day and 0.02 Pa s $^{-1}$ at the gentler US-MMS.

We had expected the taper to substitute for nudging at rough sites, and it does
not. Tapering alone leaves US-Whs 12 K too cold over a month, because the low
level advection there is at once contaminated by terrain and the real driver of
the January warming, so removing it discards the signal along with the error.
Used together with relaxation of the whole column the taper does help, bringing
the mean error at that site from 1.1 to 0.6 K, since the relaxation then has
less spurious forcing to work against. We report this because the combination
that works is not the one that the physical argument alone would suggest.

# Architecture and performance

The workflow is organized as three stages, which are shown schematically in
\autoref{fig:workflow}. Each stage is exposed as its own subcommand and can be
used on its own, so that a user who already holds ERA5 output can enter the
workflow at the second stage. A fourth subcommand chains all three together,
and the final stage can install the case file and namelist directly into an SCM
directory tree so that nothing needs to be copied by hand.

![A schematic representation of the era5-to-ccpp-scm-tool software
flow.\label{fig:workflow}](workflow.png)

ERA5 is read as a three by three stencil centered on the site of interest,
which is the smallest footprint that supports centered horizontal gradients.
The width of the stencil can be increased in whole grid cells, which trades
spatial resolution for gradients that are less sensitive to noise in the
reanalysis fields.

One property of the archive shapes the design of the extraction stage. ERA5 is
stored with one whole global field per compressed chunk, so reading a three by
three stencil costs the same as reading the entire file, which is roughly
1.3 GB per pressure level variable per day. That cost is incurred per file
rather than per site, so we provide a function that extracts many sites in a
single pass over the archive, for which the marginal cost of an additional site
is close to zero.

# Example application

To demonstrate the full workflow we generated cases at two contrasting
AmeriFlux sites: US-Whs, a semi-arid shrubland at Walnut Gulch in southern
Arizona, beginning 1 January 2019, and US-MMS, a temperate deciduous forest at
Morgan Monroe in Indiana, beginning 15 June 2019. Both cases were produced with
a single invocation of the full pipeline, contained every variable that the
reference DEPHY case file carries, and ran for a month in CCPP-SCM v7.0.0 under
the `SCM_GFS_v16` suite without further intervention. Both use relaxation of the
whole column and an advection taper of 15000 Pa, which are the settings
described above. \autoref{fig:case} shows the initial state that was written
into each case file along with the resulting simulation.

![Two cases from generation to simulation. The top row shows the initial state
written into the DEPHY case files, and the bottom row shows the month of
CCPP-SCM simulation that follows. Panel (c) tracks the column mean soil
temperature through the run. The flux panels begin at the third output step,
since starting Noah from a surface that HTESSEL equilibrated leaves a transient
over the first step. The scripts used to generate this figure are included in
the repository.\label{fig:case}](case_study.png)

The atmospheric initial state is specific to the site and season, with US-MMS
carrying roughly twice the near-surface total water of US-Whs, at 6.4 against
3.1 $g/kg$. The land surface state is likewise specific to each site. The
initial soil temperature profiles differ between the two cases in both
magnitude and in the sign of their vertical gradient. The January US-Whs profile
rises from 280 K in the uppermost layer to 291 K at two metres, as the deep soil
retains the previous summer's heat, while the June US-MMS profile falls from
292 K to 285 K over the same depth. Recovering this seasonal reversal follows
directly from taking the soil state from the reanalysis, since any single
template can supply only one of the two. The two cases also receive different
soil textures, silt loam against loam, different land cover classes, grassland
against deciduous broadleaf forest, and green vegetation fractions of 0.30 and
0.88.

Both cases track the reanalysis they were built from. Compared against ERA5 2 m
temperature over the month, the daily mean air temperature at the lowest model
level carries a bias of 0.6 K at US-Whs and 0.2 K at US-MMS, with daily root
mean square errors of 1.2 and 1.0 K and correlations of 0.95 and 0.92. We
regard this comparison, which the package produces as a diagnostic, as the
check to run before trusting a long case, since a column can remain numerically
well behaved while drifting away from anything observed.

The simulated partitioning of the surface energy budget follows from the
atmospheric and land states together. Taken over daylight hours across the
month, the evaporative fraction is 0.80 at the forest site against 0.58 at the
desert site, with mean latent heat fluxes of 270 and 76 $W / m^{2}$. Building
the same two cases on the template soil instead gives evaporative fractions of
0.90 and 0.79 and mean latent heat fluxes of 268 and 114 $W / m^{2}$. The
template carries a soil moisture saturation fraction of 0.33 at all four levels,
which is close to field capacity, together with a vegetation cover of 0.75 that
applies all year, and it therefore supplies more water to both surfaces than the
reanalysis holds at either. The effect is much the larger at the desert site,
where the template raises the evaporative fraction by 0.21 and half again as
much latent heat, and it is the site where the template is furthest from the
truth. The two sites were run in different seasons, so we present these results
as a demonstration that the tool resolves differences between sites and seasons
rather than as a controlled attribution of those differences to vegetation type.

The initial land state continues to matter for as long as we ran the cases. The
column mean soil temperature of the two US-Whs runs begins 1.8 K apart and ends
the month 3.1 K apart, and the two US-MMS runs begin 2.1 K apart and end 1.7 K
apart, so in neither case does the difference wash out. The same separation is
visible in the air above: the template-initialized runs are persistently 1 to
2 K cooler at both sites through the whole month. Initializing the land surface
from the site rather than from a template therefore changes not only the first
day of a simulation but the state the column occupies weeks later, which is the
clearest statement we can make of what the package is for.

# Limitations and future work

Land surface state is taken from the reanalysis, but three caveats attach to
the way it gets there. ERA5 describes land cover and soil texture on the same
0.25 degree grid as everything else, which is coarse next to the footprint of
an eddy covariance tower, and where a site publishes its own classification we
expect that to be used instead. The package accepts a vegetation class by its
FLUXNET abbreviation for exactly this purpose, both when a case is built and as
a patch applied to a case that has already been written. The HTESSEL and Noah
parameter tables also differ in ways that transferring the degree of saturation
reduces without removing, so soil moisture carried between them is better read
as an equivalent wetness than as a measured water content. Noah moreover begins
from a surface that was equilibrated by a different land surface model, and at
some sites this produces a transient in the surface fluxes over the first model
step, which we recommend discarding and which lengthening the spinup period
does not remove. A few fields have no ERA5 counterpart at all, among them the
maximum snow albedo, the fractional coverage weights that divide the albedo
between its strong and weak zenith angle components, and the friction velocity,
and these are still inherited from the case template [@niu2011].

A second group of limitations concerns how long a case can usefully be run, and
what has to be given up to run it that long. Left unconstrained the column
drifts, so a run of more than about a week needs the relaxation described in the
methods, and the setting that proved robust at both of our sites relaxes the
whole column rather than the free troposphere alone.
The boundary layer is how the land surface communicates with
the atmosphere, and relaxing it pulls near-surface temperature and humidity
toward the reanalysis, so a study of how strongly the surface controls the air
above it may understate the coupling. However, fluxes are still diagnosed from the land state, the soil still evolves freely, and the separation between our two soil initializations persists
for the whole month. A user whose question is about the boundary layer rather
than about the surface should prefer relaxing only the free troposphere, accept
a shorter run, and check the result against the reanalysis before trusting it.

The advection taper carries its own caveat. It exists because horizontal
gradients taken close to the ground describe terrain rather than flow, but at a
site with pronounced relief that same low level forcing is also carrying the
real synoptic signal, and the taper cannot separate the two. Tapering therefore
removes real advection along with the spurious part, which is why we use it
alongside relaxation rather than in place of it. Neither control is a
substitute for the underlying difficulty, which is that a pressure level
framework describes the atmosphere near steep terrain poorly. Sites in flat
country need neither control for runs of a few days.

Two further caveats follow from the source data. The radiative heating rate
that we diagnose is a bulk column mean value, because ERA5 archives radiative
fluxes only at the top of the atmosphere and at the surface, and the vertical
structure of the heating cannot be recovered from boundary values alone. The
cases that we write therefore enable the internal radiation scheme of the SCM,
which computes radiation directly and ignores the prescribed rate. Geostrophic
winds are also undefined at the equator and unreliable near it, so cases
generated in the deep tropics should be treated with caution.

# Acknowledgements

We thank the NSF National Center for Atmospheric Research and the Research Data
Archive for hosting and maintaining the public ERA5 archive, and the
Developmental Testbed Center for the CCPP Single Column Model. Funding for
this work was provided by the NSF WPO program under award <<<PLACEHOLDER>>>

# References
