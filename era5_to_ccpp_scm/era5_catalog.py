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

#: Time-invariant fields, useful for site metadata (surface elevation, mask).
INVARIANT_VARIABLES: Dict[str, Era5Variable] = {
    "z_sfc": Era5Variable("128_129_z", "ll025sc", STREAM_INVARIANT, "Z"),
    "lsm": Era5Variable("128_172_lsm", "ll025sc", STREAM_INVARIANT, "LSM"),
}


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
