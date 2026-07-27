"""Archive naming: filename span parsing and month enumeration."""

import datetime as dt

import pytest

from era5_to_ccpp_scm.era5_catalog import (
    PRESSURE_VARIABLES,
    RADIATION_VARIABLES,
    SURFACE_VARIABLES,
    months_between,
    parse_span,
)


class TestParseSpan:
    def test_parses_a_daily_pressure_level_name(self):
        name = ("e5.oper.an.pl.128_130_t.ll025sc."
                "2019010100_2019010123.nc")
        assert parse_span(name) == (
            dt.datetime(2019, 1, 1, 0),
            dt.datetime(2019, 1, 1, 23),
        )

    def test_parses_a_monthly_surface_name(self):
        name = ("e5.oper.an.sfc.128_134_sp.ll025sc."
                "2019010100_2019013123.nc")
        assert parse_span(name) == (
            dt.datetime(2019, 1, 1, 0),
            dt.datetime(2019, 1, 31, 23),
        )

    def test_half_month_accumulation_spills_into_the_next_month(self):
        # The second half-month file of January ends on 1 February at 06Z.
        name = ("e5.oper.fc.sfc.accumu.128_176_ssr.ll025sc."
                "2019011606_2019020106.nc")
        start, end = parse_span(name)
        assert start.month == 1
        assert end == dt.datetime(2019, 2, 1, 6)

    @pytest.mark.parametrize("name", [
        "index.html",
        "e5.oper.an.pl.128_130_t.ll025sc.nc",
        "README",
        "e5.oper.an.pl.128_130_t.ll025sc.2019010100_20190101.nc",
    ])
    def test_returns_none_for_names_without_a_span(self, name):
        assert parse_span(name) is None


class TestMonthsBetween:
    def test_single_month(self):
        assert months_between(dt.datetime(2019, 1, 1),
                              dt.datetime(2019, 1, 31)) == ["201901"]

    def test_inclusive_of_both_endpoints(self):
        assert months_between(dt.datetime(2019, 1, 15),
                              dt.datetime(2019, 3, 2)) == [
            "201901", "201902", "201903",
        ]

    def test_crosses_a_year_boundary(self):
        assert months_between(dt.datetime(2018, 11, 30),
                              dt.datetime(2019, 2, 1)) == [
            "201811", "201812", "201901", "201902",
        ]

    def test_end_before_start_yields_nothing(self):
        assert months_between(dt.datetime(2019, 6, 1),
                              dt.datetime(2019, 5, 1)) == []


class TestVariableTables:
    def test_pressure_variables_cover_the_forcing_calculation(self):
        # era5_to_scm_forcing reads exactly these off the pressure-level file.
        assert set(PRESSURE_VARIABLES) == {"z", "t", "u", "v", "q", "w"}

    def test_wind_components_use_the_uv_grid_tag(self):
        assert PRESSURE_VARIABLES["u"].grid == "ll025uv"
        assert PRESSURE_VARIABLES["v"].grid == "ll025uv"
        assert PRESSURE_VARIABLES["t"].grid == "ll025sc"

    def test_surface_table_supplies_the_fields_the_conversion_needs(self):
        assert {"sp", "t2m"} <= set(SURFACE_VARIABLES)

    def test_radiation_table_supplies_the_four_net_fluxes(self):
        assert {"ssr", "str", "tsr", "ttr"} <= set(RADIATION_VARIABLES)

    def test_netcdf_names_avoid_a_leading_digit(self):
        # The RDA archive prefixes names that would otherwise start with a
        # digit, which is not a legal NetCDF identifier.
        for table in (PRESSURE_VARIABLES, SURFACE_VARIABLES,
                      RADIATION_VARIABLES):
            for variable in table.values():
                assert not variable.nc_name[0].isdigit()
