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
  - name: Andrew R. Bennett
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
itself [@randall1996]. Because a single column integrates in seconds rather
than hours, SCMs are widely used for developing and testing parameterizations,
and for examining what a physics suite does at a particular place and time.
Running an SCM requires a case, which consists of initial profiles of the
atmospheric state, large-scale advective tendencies of heat and moisture,
vertical motion, geostrophic winds, and surface and land-surface properties,
all assembled in the layout that the target model reads. Building such a case
has traditionally been an ad hoc, per-study effort, and as a result the case
libraries that the community shares consist of a few dozen curated field
campaign cases.

`era5-to-ccpp-scm-tool` is a Python package and command line tool that builds
SCM cases from ERA5 reanalysis [@hersbach2020] for any location on the globe.
Given a latitude, longitude, and date range, it extracts the reanalysis fields
that are needed, derives the full set of large-scale forcing terms, and writes
a case in the DEPHY format [@dephy] along with its configuration namelist, so
that the case can be run directly in the Common Community Physics Package
Single Column Model (CCPP-SCM) [@heinzeller2023; @ccppscm]. A complete case is
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
and API access as a Python module.

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
reproduces exactly on a high-performance computing system. We also noticed that
the forcing derivations were usually implemented as ad hoc solutions based on
similar sets of equations, and wanted to formalize the process by developing
this package. Each individual step is straightforward, but each is also easy to
get subtly wrong, whether in the sign of a meridional gradient computed on the
north-first latitude axis that ERA5 uses, in a metric factor, or in a unit
conversion. Packaging the derivations together with a test suite makes the
conventions explicit and the results comparable across studies.

We chose CCPP-SCM as the target model because of its lineage. CCPP-SCM is the
single-column driver for the Common Community Physics Package
[@heinzeller2023], and the physics suites that it runs are the same source
code, invoked through the same framework, as the physics in NOAA's operational
Unified Forecast System. A parameterization that is examined in a column is
therefore the one that runs in the forecast model, and there is no separate
single-column implementation that must be kept in sync. Suites are selected at
run time from suite definition files, which makes it inexpensive to push a
single case through several physics configurations and to attribute the
differences to the physics rather than to the setup. CCPP-SCM also couples a
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
observational constraint and that temporal homogeneity are the properties that
matter, and neither is available from a raw forecast archive. We consider the
two tools to be complementary rather than competing.

# Methods

ERA5 and a DEPHY case file differ in nearly every convention, and the
conversions between them fall into five broad groups.

The first group concerns coordinates and indexing. ERA5 latitudes are stored
north first, so a northward gradient is computed as the difference between the
first and third rows of the stencil rather than the reverse. Longitudes run
from 0 to 360 degrees and are unwrapped when a site lies near the prime
meridian, and pressure levels arrive in hPa and are re-indexed onto their
converted values rather than simply rescaled.

The second group concerns time. The accumulated radiative fluxes are archived
as hourly totals in J m$^{-2}$ and are divided by the accumulation period to
give rates in W m$^{-2}$. Those fluxes come from a forecast stream dimensioned
by forecast initial time and forecast hour, which is flattened onto valid time
to give a continuous hourly series. The absolute timestamps that ERA5 carries
are then converted to seconds since the start of the case, which is the
convention DEPHY uses.

The third group is thermodynamic. Temperature and specific humidity are
converted to liquid water potential temperature and total water specific
humidity, the pressure velocity that ERA5 archives is converted to a vertical
velocity in m s$^{-1}$ using the thermodynamic relations provided by MetPy, and
geopotential is converted to geopotential height.

The fourth group is dynamical. Geostrophic winds are computed from the
gradients of geopotential height across the stencil, and the horizontal and
vertical advective tendencies of liquid water potential temperature and total
water are computed using three point derivatives on a non-uniform stencil with
spherical metric factors applied. A column mean radiative heating rate is
diagnosed from the difference between the net radiative fluxes at the top of
the atmosphere and at the surface.

The final group concerns the schema. ERA5 short names are mapped onto DEPHY
variable names, and the grouped arrays that older CCPP-SCM releases used are
written into the flat layout that DEPHY specifies. This conversion is not
purely cosmetic, because DEPHY encodes which forcing terms are active in the
global attributes of the case file. The attributes are set to enable advection
of liquid water potential temperature and total water, to supply pressure
velocity and geostrophic forcing, to disable nudging, and to enable the
internal radiation scheme. A case whose attributes disagree with the arrays it
carries will run to completion and produce a plausible but incorrect answer,
which is one of the reasons we consider it worthwhile to package these steps
together. The companion Fortran namelist that declares the case to the SCM is
written at the same stage.

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
is close to zero. We distribute work across processes rather than threads,
because the HDF5 library underlying both of the NetCDF engines we support is
not thread safe and concurrent reads within a single process were not stable.
Reading a two day extraction from the filesystem backend took 248 s in serial
and 48 s when parallelized in this way.

# Example application

To demonstrate the full workflow we generated cases at two contrasting
AmeriFlux sites: US-Whs, a semi-arid shrubland at Walnut Gulch in southern
Arizona, for 1 January 2019, and US-MMS, a temperate deciduous forest at Morgan
Monroe in Indiana, for 15 June 2019. Both cases were produced with a single
invocation of the full pipeline, contained every variable that the reference
DEPHY case file carries, and ran for 24 hours in CCPP-SCM v7.0.0 under the
`SCM_GFS_v16` suite without further intervention. \autoref{fig:case} shows the
initial state that was written into each case file along with the resulting
simulation.

![Two cases from generation to simulation. The top row shows the initial state
written into the DEPHY case files, and the bottom row shows the resulting
CCPP-SCM simulations. The scripts used to generate this figure are included in
the repository.\label{fig:case}](case_study.png)

The atmospheric initial state is specific to the site and season, with US-MMS
carrying nearly three times the near-surface total water of US-Whs, at 8.6
against 3.2 g kg$^{-1}$. The simulated partitioning of the surface energy
budget follows from this. The daily mean evaporative fraction is 0.76 at the
forest site against 0.29 at the desert site, peak latent heat fluxes are 447
and 106 W m$^{-2}$ respectively, and the deeper and more variable daytime
boundary layer that the model produces at US-MMS is consistent with that
partitioning.

The initial soil temperature profiles shown in the third panel are identical
between the two cases, because soil state is currently inherited from the
GABLS3 template rather than derived at the site, as discussed below. The
atmospheric contrast in this figure is therefore real, while a soil contrast
would not be. The sensible heat flux above 300 W m$^{-2}$ that the January
US-Whs case produces near midday is higher than we would expect for a desert in
winter, and is plausibly an artifact of a warm template soil. The two sites
were also run in different seasons, so we present these results as a
demonstration that the tool resolves differences between sites and seasons
rather than as a controlled attribution of those differences to vegetation
type.

# Limitations and future work

Soil and land surface state, which includes soil temperature and moisture,
vegetation and soil type, and albedo, is currently inherited from the GABLS3
template rather than taken from the site, because the land surface fields that
ERA5 archives are not on the four layer Noah-MP soil grid that the model
expects [@niu2011]. This is the most consequential limitation of the package
for land-atmosphere work, and until it is addressed users should override these
fields for each site before drawing conclusions about surface fluxes. We plan
to add site-specific soil initialization in a subsequent release, and we
consider it the most valuable contribution that others could make.

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
