"""Shared fixtures.

Every test in this suite is network-free: the only inputs are a small ERA5
extraction committed under ``casegen_walnut_gulch/data`` and arrays built in
memory.
"""

import os

import pytest

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "casegen_walnut_gulch",
    "data",
)

#: Two days of ERA5 over Walnut Gulch (US-Whs), 3x3 stencil, 37 levels.
#: These files predate the NSF NCAR downloader, so they also exercise the
#: CDS-style names (``valid_time``, ``pressure_level``, hPa levels) and the
#: separate radiation file that the current downloader no longer writes.
WALNUT_SFC = os.path.join(DATA_DIR, "walnut_gulch_sfc.nc")
WALNUT_PL = os.path.join(DATA_DIR, "walnut_gulch_pl.nc")
WALNUT_RAD = os.path.join(DATA_DIR, "walnut_gulch_rad.nc")


@pytest.fixture(scope="session")
def walnut_files():
    """Paths to the committed Walnut Gulch extraction."""
    missing = [p for p in (WALNUT_SFC, WALNUT_PL, WALNUT_RAD)
               if not os.path.exists(p)]
    if missing:
        pytest.skip(f"test data not present: {missing}")
    return WALNUT_SFC, WALNUT_PL, WALNUT_RAD


@pytest.fixture(scope="session")
def walnut_forcings(walnut_files):
    """Processed SCM forcings for Walnut Gulch, built once for the session."""
    from era5_to_ccpp_scm.cli import _core_convert_forcings

    sfc, pl, rad = walnut_files
    return _core_convert_forcings(sfc, pl, None, rad)
