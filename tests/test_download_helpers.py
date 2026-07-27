"""Grid indexing and date parsing in the downloader.

These are the parts of ``download_era5`` that can be exercised without
touching the archive.
"""

import datetime as dt

import numpy as np
import pytest
import xarray as xr

from era5_to_ccpp_scm.download_era5 import (
    _GRID_STEP,
    _N_LAT,
    _N_LON,
    _parse_date,
    _stencil_indices,
    _unwrap_longitude,
)


class TestStencilIndices:
    def test_centre_lands_on_the_nearest_grid_point(self):
        # Walnut Gulch: 31.7438N, 110.0522W -> nearest 0.25 deg cell.
        lat_idx, lon_idx = _stencil_indices(31.7438, -110.0522)
        assert 90.0 - lat_idx[1] * _GRID_STEP == pytest.approx(31.75)
        assert lon_idx[1] * _GRID_STEP == pytest.approx(250.0)

    def test_returns_three_points_per_axis(self):
        lat_idx, lon_idx = _stencil_indices(40.0, 10.0)
        assert len(lat_idx) == 3 and len(lon_idx) == 3

    def test_latitude_indices_increase_southward(self):
        lat_idx, _ = _stencil_indices(40.0, 10.0)
        assert lat_idx == [lat_idx[1] - 1, lat_idx[1], lat_idx[1] + 1]

    def test_longitude_wraps_across_the_prime_meridian(self):
        _, lon_idx = _stencil_indices(0.5, 0.0)
        assert lon_idx == [_N_LON - 1, 0, 1]

    def test_longitude_wraps_at_the_end_of_the_grid(self):
        _, lon_idx = _stencil_indices(0.5, 359.75)
        assert lon_idx == [_N_LON - 2, _N_LON - 1, 0]

    def test_negative_and_positive_longitude_conventions_agree(self):
        assert _stencil_indices(45.0, -110.0) == _stencil_indices(45.0, 250.0)

    @pytest.mark.parametrize("lat", [90.0, 89.9, -90.0, -89.9])
    def test_stencil_stays_on_the_grid_at_the_poles(self, lat):
        lat_idx, _ = _stencil_indices(lat, 0.0)
        assert min(lat_idx) >= 0
        assert max(lat_idx) <= _N_LAT - 1

    def test_a_wider_step_widens_the_stencil_but_keeps_it_3x3(self):
        lat_idx, lon_idx = _stencil_indices(40.0, 10.0, step=4)
        assert len(lat_idx) == 3 and len(lon_idx) == 3
        assert lat_idx[2] - lat_idx[0] == 8
        assert lon_idx[2] - lon_idx[0] == 8

    def test_a_wide_step_still_clamps_at_the_pole(self):
        lat_idx, _ = _stencil_indices(90.0, 0.0, step=5)
        assert min(lat_idx) >= 0

    @pytest.mark.parametrize("step", [0, -1])
    def test_rejects_a_non_positive_step(self, step):
        with pytest.raises(ValueError, match="positive"):
            _stencil_indices(40.0, 10.0, step=step)

    @pytest.mark.parametrize("lat", [90.5, -91.0])
    def test_rejects_an_out_of_range_latitude(self, lat):
        with pytest.raises(ValueError, match=r"\[-90, 90\]"):
            _stencil_indices(lat, 10.0)


class TestParseDate:
    def test_bare_date_starts_at_midnight(self):
        assert _parse_date("2019-01-01") == dt.datetime(2019, 1, 1, 0)

    def test_bare_date_as_a_range_end_means_the_end_of_that_day(self):
        # A one-day request must cover 00Z through 23Z, not a single hour.
        assert _parse_date("2019-01-01", end_of_day=True) == \
            dt.datetime(2019, 1, 1, 23)

    def test_explicit_time_is_not_promoted_to_the_end_of_the_day(self):
        assert _parse_date("2019-01-01 06:00:00", end_of_day=True) == \
            dt.datetime(2019, 1, 1, 6)

    @pytest.mark.parametrize("text,expected", [
        ("2019-06-15 12:00:00", dt.datetime(2019, 6, 15, 12)),
        ("2019-06-15T12:00:00", dt.datetime(2019, 6, 15, 12)),
        ("2019-06-15 12:00", dt.datetime(2019, 6, 15, 12)),
        ("  2019-06-15  ", dt.datetime(2019, 6, 15, 0)),
    ])
    def test_accepts_the_documented_formats(self, text, expected):
        assert _parse_date(text) == expected

    def test_passes_a_datetime_through_unchanged(self):
        stamp = dt.datetime(2019, 3, 4, 5)
        assert _parse_date(stamp) is stamp

    def test_a_date_object_honours_end_of_day(self):
        assert _parse_date(dt.date(2019, 3, 4), end_of_day=True) == \
            dt.datetime(2019, 3, 4, 23)

    def test_rejects_an_unparseable_string(self):
        with pytest.raises(ValueError, match="Cannot parse date"):
            _parse_date("01/02/2019")


class TestUnwrapLongitude:
    def test_makes_a_prime_meridian_stencil_monotonic(self):
        da = xr.DataArray(
            np.zeros(3), dims=["longitude"],
            coords={"longitude": [359.75, 0.0, 0.25]},
        )
        lons = _unwrap_longitude(da)["longitude"].values
        # Spacing must stay 0.25 deg so centred differences are right.
        assert np.allclose(np.diff(lons), _GRID_STEP)

    def test_leaves_an_ordinary_stencil_alone(self):
        da = xr.DataArray(
            np.zeros(3), dims=["longitude"],
            coords={"longitude": [249.75, 250.0, 250.25]},
        )
        lons = _unwrap_longitude(da)["longitude"].values
        assert np.allclose(lons, [249.75, 250.0, 250.25])
