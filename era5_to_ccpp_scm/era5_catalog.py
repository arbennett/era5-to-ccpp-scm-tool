"""
Catalogue of the NSF NCAR ERA5 archive (RDA ds633.0).

The archive is published in two places with an identical directory layout:

  * ``s3://nsf-ncar-era5``                              (public, anonymous)
  * ``/glade/campaign/collections/rda/data/d633000``    (NCAR Glade)

Files are named::

    {stream}/{YYYYMM}/{stream}.{table}_{param}_{abbr}.{grid}.{start}_{end}.nc

where ``start``/``end`` are ``YYYYMMDDHH`` stamps delimiting the *inclusive*
time span held in the file.  Each stream packages its time span differently:

  ``e5.oper.an.pl``          one file per variable per **day**
  ``e5.oper.an.sfc``         one file per variable per **month**
  ``e5.oper.fc.sfc.accumu``  one file per variable per **half month**
  ``e5.oper.invariant``      a single file for the whole record

This module holds only the naming knowledge.  Actual I/O lives in
``download_era5.py``.
"""

from __future__ import annotations

import datetime as dt
import re
from typing import Dict, NamedTuple

# ---------------------------------------------------------------------------
# Archive roots
# ---------------------------------------------------------------------------

S3_BUCKET = "nsf-ncar-era5"
S3_URL = f"https://{S3_BUCKET}.s3.amazonaws.com"
GLADE_ROOT = "/glade/campaign/collections/rda/data/d633000"


# ---------------------------------------------------------------------------
# Streams
# ---------------------------------------------------------------------------

STREAM_PRESSURE = "e5.oper.an.pl"
STREAM_SURFACE = "e5.oper.an.sfc"
STREAM_ACCUM = "e5.oper.fc.sfc.accumu"
STREAM_INVARIANT = "e5.oper.invariant"


class Era5Variable(NamedTuple):
    """One ERA5 variable as it is stored in the archive.

    Attributes
    ----------
    code : str
        The ``{table}_{param}_{abbr}`` token embedded in the filename.
    grid : str
        Grid tag; ``ll025sc`` for scalars, ``ll025uv`` for wind components.
    stream : str
        Which archive stream holds the variable.
    nc_name : str
        The variable name *inside* the NetCDF file (ERA5 RDA uppercases these,
        and prefixes names that would otherwise start with a digit).
    """

    code: str
    grid: str
    stream: str
    nc_name: str


#: Pressure-level analysis variables required to build SCM forcings.
#: Keys are the CDS-style short names the conversion code expects.
PRESSURE_VARIABLES: Dict[str, Era5Variable] = {
    "z": Era5Variable("128_129_z", "ll025sc", STREAM_PRESSURE, "Z"),
    "t": Era5Variable("128_130_t", "ll025sc", STREAM_PRESSURE, "T"),
    "u": Era5Variable("128_131_u", "ll025uv", STREAM_PRESSURE, "U"),
    "v": Era5Variable("128_132_v", "ll025uv", STREAM_PRESSURE, "V"),
    "q": Era5Variable("128_133_q", "ll025sc", STREAM_PRESSURE, "Q"),
    "w": Era5Variable("128_135_w", "ll025sc", STREAM_PRESSURE, "W"),
}

#: Instantaneous surface analysis variables.
SURFACE_VARIABLES: Dict[str, Era5Variable] = {
    "sp": Era5Variable("128_134_sp", "ll025sc", STREAM_SURFACE, "SP"),
    "t2m": Era5Variable("128_167_2t", "ll025sc", STREAM_SURFACE, "VAR_2T"),
    "d2m": Era5Variable("128_168_2d", "ll025sc", STREAM_SURFACE, "VAR_2D"),
    "u10": Era5Variable("128_165_10u", "ll025sc", STREAM_SURFACE, "VAR_10U"),
    "v10": Era5Variable("128_166_10v", "ll025sc", STREAM_SURFACE, "VAR_10V"),
    "skt": Era5Variable("128_235_skt", "ll025sc", STREAM_SURFACE, "SKT"),
}

#: Accumulated radiative fluxes from the forecast stream.  These are stored as
#: hourly accumulations in J m-2 and are converted to W m-2 on read.
RADIATION_VARIABLES: Dict[str, Era5Variable] = {
    "ssr": Era5Variable("128_176_ssr", "ll025sc", STREAM_ACCUM, "SSR"),
    "str": Era5Variable("128_177_str", "ll025sc", STREAM_ACCUM, "STR"),
    "tsr": Era5Variable("128_178_tsr", "ll025sc", STREAM_ACCUM, "TSR"),
    "ttr": Era5Variable("128_179_ttr", "ll025sc", STREAM_ACCUM, "TTR"),
    "ssrd": Era5Variable("128_169_ssrd", "ll025sc", STREAM_ACCUM, "SSRD"),
    "strd": Era5Variable("128_175_strd", "ll025sc", STREAM_ACCUM, "STRD"),
}

#: Land-surface analysis variables used to initialise the LSM.  These share the
#: surface analysis stream with :data:`SURFACE_VARIABLES` and are read by the
#: same code path; they are kept separate only so the download can be skipped.
#:
#: ERA5's land surface is HTESSEL, whose four soil layers and soil/vegetation
#: parameter tables differ from Noah's.  Translating these onto the Noah grid
#: and parameter set is the job of :mod:`era5_to_ccpp_scm.land_state`.
LAND_VARIABLES: Dict[str, Era5Variable] = {
    # Soil temperature, HTESSEL layers 0-7, 7-28, 28-100, 100-289 cm
    "stl1": Era5Variable("128_139_stl1", "ll025sc", STREAM_SURFACE, "STL1"),
    "stl2": Era5Variable("128_170_stl2", "ll025sc", STREAM_SURFACE, "STL2"),
    "stl3": Era5Variable("128_183_stl3", "ll025sc", STREAM_SURFACE, "STL3"),
    "stl4": Era5Variable("128_236_stl4", "ll025sc", STREAM_SURFACE, "STL4"),
    # Volumetric soil water on the same four layers, m3 m-3
    "swvl1": Era5Variable("128_039_swvl1", "ll025sc", STREAM_SURFACE, "SWVL1"),
    "swvl2": Era5Variable("128_040_swvl2", "ll025sc", STREAM_SURFACE, "SWVL2"),
    "swvl3": Era5Variable("128_041_swvl3", "ll025sc", STREAM_SURFACE, "SWVL3"),
    "swvl4": Era5Variable("128_042_swvl4", "ll025sc", STREAM_SURFACE, "SWVL4"),
    # Snow: water equivalent depth (m), density (kg m-3), albedo, temperature
    "sd": Era5Variable("128_141_sd", "ll025sc", STREAM_SURFACE, "SD"),
    "rsn": Era5Variable("128_033_rsn", "ll025sc", STREAM_SURFACE, "RSN"),
    "asn": Era5Variable("128_032_asn", "ll025sc", STREAM_SURFACE, "ASN"),
    "tsn": Era5Variable("128_238_tsn", "ll025sc", STREAM_SURFACE, "TSN"),
    # Skin reservoir content: canopy-intercepted water, m of water equivalent
    "src": Era5Variable("128_198_src", "ll025sc", STREAM_SURFACE, "SRC"),
    # Albedo, direct ("parallel") and diffuse, in UV/visible and near infrared.
    # This quartet maps one-to-one onto the GFS strong/weak cosz pairs.
    "aluvp": Era5Variable("128_015_aluvp", "ll025sc", STREAM_SURFACE, "ALUVP"),
    "aluvd": Era5Variable("128_016_aluvd", "ll025sc", STREAM_SURFACE, "ALUVD"),
    "alnip": Era5Variable("128_017_alnip", "ll025sc", STREAM_SURFACE, "ALNIP"),
    "alnid": Era5Variable("128_018_alnid", "ll025sc", STREAM_SURFACE, "ALNID"),
    # Surface roughness length, m
    "fsr": Era5Variable("128_244_fsr", "ll025sc", STREAM_SURFACE, "FSR"),
    # Leaf area index of low and high vegetation, m2 m-2
    "lailv": Era5Variable("128_066_lailv", "ll025sc", STREAM_SURFACE, "LAILV"),
    "laihv": Era5Variable("128_067_laihv", "ll025sc", STREAM_SURFACE, "LAIHV"),
}

#: Time-invariant fields: site metadata (surface elevation, mask) and the
#: land-cover and soil-texture classes needed to initialise the LSM.
INVARIANT_VARIABLES: Dict[str, Era5Variable] = {
    "z_sfc": Era5Variable("128_129_z", "ll025sc", STREAM_INVARIANT, "Z"),
    "lsm": Era5Variable("128_172_lsm", "ll025sc", STREAM_INVARIANT, "LSM"),
    "cvl": Era5Variable("128_027_cvl", "ll025sc", STREAM_INVARIANT, "CVL"),
    "cvh": Era5Variable("128_028_cvh", "ll025sc", STREAM_INVARIANT, "CVH"),
    "tvl": Era5Variable("128_029_tvl", "ll025sc", STREAM_INVARIANT, "TVL"),
    "tvh": Era5Variable("128_030_tvh", "ll025sc", STREAM_INVARIANT, "TVH"),
    "slt": Era5Variable("128_043_slt", "ll025sc", STREAM_INVARIANT, "SLT"),
    "slor": Era5Variable("128_163_slor", "ll025sc", STREAM_INVARIANT, "SLOR"),
}

#: The invariant stream holds one file for the entire record, stamped with the
#: start of the ERA5 period rather than with the span it applies to.  Selecting
#: it by date would reject it for every case after January 1979.
INVARIANT_YYYYMM = "197901"


#: Filenames encode their span as ``YYYYMMDDHH_YYYYMMDDHH`` before ``.nc``.
_SPAN_RE = re.compile(r"\.(\d{10})_(\d{10})\.nc$")


def parse_span(filename: str):
    """Return the ``(start, end)`` datetimes encoded in an archive filename.

    Returns ``None`` when the name does not carry a recognisable span, which
    lets callers skip index pages and other non-data keys.
    """
    match = _SPAN_RE.search(filename)
    if match is None:
        return None
    fmt = "%Y%m%d%H"
    return (
        dt.datetime.strptime(match.group(1), fmt),
        dt.datetime.strptime(match.group(2), fmt),
    )


def months_between(start: dt.datetime, end: dt.datetime):
    """Yield ``YYYYMM`` strings covering ``start`` through ``end`` inclusive.

    The accumulated-flux stream files its final half-month under the *starting*
    month even though the data run into the following month, so callers that
    need those files should extend ``end`` by a day before calling this.
    """
    months = []
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        months.append(f"{year:04d}{month:02d}")
        month += 1
        if month > 12:
            month = 1
            year += 1
    return months
