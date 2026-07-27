"""Forcing derivation: gradients, advection, geostrophy, radiative heating."""

import numpy as np
import pytest
import xarray as xr

from era5_to_ccpp_scm.convert_forcing import (
    _CP_DRY,
    _GRAVITY,
    _as_flux_rate,
    _three_point_first_derivative,
    calculate_advection,
    calculate_geostrophic_wind,
    calculate_grid_spacing,
    calculate_horizontal_gradients,
    calculate_radiative_heating,
    era5_to_scm_forcing,
)

R_EARTH = 6_371_000.0


def scalar(da):
    """The single value held by a DataArray that kept its length-1 dims."""
    values = np.asarray(da.values).ravel()
    assert values.size == 1, f"expected one value, got {values.size}"
    return float(values[0])

#: ERA5 latitudes run north-first; longitudes increase.
LATS = np.array([31.75 + 0.25, 31.75, 31.75 - 0.25])
LONS = np.array([249.75, 250.0, 250.25])


class TestThreePointDerivative:
    def test_exact_for_a_linear_function(self):
        x0, x1, x2 = 0.0, 1.0, 3.0
        slope = 2.5
        f = [slope * x for x in (x0, x1, x2)]
        assert _three_point_first_derivative(*f, x0, x1, x2) == \
            pytest.approx(slope)

    def test_exact_for_a_quadratic_on_a_non_uniform_stencil(self):
        # The three-point formula is second-order, so a quadratic is exact.
        x0, x1, x2 = 0.0, 1.0, 4.0
        f = [x ** 2 for x in (x0, x1, x2)]
        assert _three_point_first_derivative(*f, x0, x1, x2) == \
            pytest.approx(2 * x1)

    def test_reduces_to_a_centred_difference_when_evenly_spaced(self):
        x0, x1, x2 = -1.0, 0.0, 1.0
        f0, f1, f2 = 3.0, 7.0, 11.0
        assert _three_point_first_derivative(f0, f1, f2, x0, x1, x2) == \
            pytest.approx((f2 - f0) / 2.0)

    def test_broadcasts_over_array_valued_samples(self):
        f0 = np.zeros((2, 3))
        f1 = np.ones((2, 3))
        f2 = np.full((2, 3), 2.0)
        out = _three_point_first_derivative(f0, f1, f2, 0.0, 1.0, 2.0)
        assert out.shape == (2, 3)
        assert np.allclose(out, 1.0)

    @pytest.mark.parametrize("xs", [(0.0, 0.0, 1.0), (0.0, 1.0, 1.0),
                                    (2.0, 1.0, 0.0)])
    def test_rejects_a_non_increasing_coordinate(self, xs):
        with pytest.raises(ValueError, match="strictly increasing"):
            _three_point_first_derivative(1.0, 2.0, 3.0, *xs)


class TestGridSpacing:
    def test_meridional_spacing_is_independent_of_latitude(self):
        _, dy = calculate_grid_spacing(LATS, LONS)
        assert dy == pytest.approx(R_EARTH * np.deg2rad(0.25), rel=1e-6)

    def test_zonal_spacing_shrinks_with_the_cosine_of_latitude(self):
        dx_low, _ = calculate_grid_spacing(np.array([0.25, 0.0, -0.25]), LONS)
        dx_high, _ = calculate_grid_spacing(np.array([60.25, 60.0, 59.75]), LONS)
        assert dx_high == pytest.approx(dx_low * np.cos(np.deg2rad(60.0)),
                                        rel=1e-6)


class TestHorizontalGradients:
    def _patch(self, values):
        return xr.DataArray(
            np.asarray(values, dtype=float)[None, None, :, :],
            dims=("time", "levels", "latitude", "longitude"),
            coords={"latitude": LATS, "longitude": LONS},
        )

    def test_uniform_field_has_no_gradient(self):
        d_dx, d_dy = calculate_horizontal_gradients(
            self._patch(np.ones((3, 3))), LATS, LONS)
        assert scalar(d_dx) == pytest.approx(0.0)
        assert scalar(d_dy) == pytest.approx(0.0)

    def test_zonal_gradient_is_positive_when_the_field_increases_eastward(self):
        field = np.tile(np.array([0.0, 1.0, 2.0]), (3, 1))  # varies with lon
        d_dx, d_dy = calculate_horizontal_gradients(
            self._patch(field), LATS, LONS)
        dx, _ = calculate_grid_spacing(LATS, LONS)
        assert scalar(d_dx) == pytest.approx(2.0 / (2 * dx))
        assert scalar(d_dy) == pytest.approx(0.0)

    def test_meridional_gradient_is_positive_when_the_field_increases_northward(self):
        # Row 0 is the northernmost row in ERA5's descending latitude order.
        field = np.tile(np.array([[2.0], [1.0], [0.0]]), (1, 3))
        _, d_dy = calculate_horizontal_gradients(
            self._patch(field), LATS, LONS)
        _, dy = calculate_grid_spacing(LATS, LONS)
        assert scalar(d_dy) == pytest.approx(2.0 / (2 * dy))


class TestGeostrophicWind:
    def _height_patch(self, values):
        return xr.DataArray(
            np.asarray(values, dtype=float)[None, None, :, :],
            dims=("time", "levels", "latitude", "longitude"),
            coords={"latitude": LATS, "longitude": LONS},
        )

    def test_flat_geopotential_gives_no_geostrophic_wind(self):
        u_g, v_g = calculate_geostrophic_wind(
            self._height_patch(np.full((3, 3), 5000.0)), LATS, LONS)
        assert scalar(u_g) == pytest.approx(0.0)
        assert scalar(v_g) == pytest.approx(0.0)

    def test_height_falling_northward_drives_a_westerly_in_the_north(self):
        # Northern hemisphere, height decreasing toward the pole: geostrophic
        # balance gives eastward (positive u_g) flow.
        field = np.tile(np.array([[0.0], [10.0], [20.0]]), (1, 3))
        u_g, _ = calculate_geostrophic_wind(
            self._height_patch(field), LATS, LONS)
        assert scalar(u_g) > 0.0

    def test_rejects_the_equator_where_the_coriolis_parameter_vanishes(self):
        equator = np.array([0.25, 0.0, -0.25])
        with pytest.raises(ValueError, match="Coriolis"):
            calculate_geostrophic_wind(
                self._height_patch(np.zeros((3, 3))), equator, LONS)


class TestAdvection:
    #: Pressure in Pa, ascending as the conversion code requires.
    LEVELS = np.array([50_000.0, 70_000.0, 85_000.0])

    def _uniform(self, value, shape=(1, 3, 3, 3)):
        return np.full(shape, float(value))

    def test_uniform_field_produces_no_advection(self):
        var = self._uniform(300.0)
        h, v = calculate_advection(
            var, self._uniform(10.0), self._uniform(5.0), self._uniform(0.1),
            LATS, LONS, self.LEVELS)
        assert np.allclose(h, 0.0)
        assert np.allclose(v, 0.0)

    def test_horizontal_advection_opposes_the_upstream_gradient(self):
        # Field increasing eastward, wind blowing eastward: the column cools
        # (negative tendency) because colder air is arriving from upstream.
        var = np.zeros((1, 3, 3, 3))
        var[..., :] = np.array([0.0, 1.0, 2.0])  # varies along longitude
        u = self._uniform(10.0)
        h, _ = calculate_advection(
            var, u, self._uniform(0.0), self._uniform(0.0),
            LATS, LONS, self.LEVELS)
        assert np.all(h < 0.0)

    def test_horizontal_advection_matches_minus_u_times_the_gradient(self):
        var = np.zeros((1, 3, 3, 3))
        var[..., :] = np.array([0.0, 1.0, 2.0])
        u_value = 10.0
        h, _ = calculate_advection(
            var, self._uniform(u_value), self._uniform(0.0),
            self._uniform(0.0), LATS, LONS, self.LEVELS)

        dx = R_EARTH * np.cos(np.deg2rad(LATS[1])) * np.deg2rad(0.25)
        assert np.allclose(h, -u_value * (1.0 / dx), rtol=1e-6)

    def test_vertical_advection_uses_the_pressure_derivative(self):
        # Linear in pressure: d(var)/dp is exactly the slope everywhere.
        slope = 1e-4
        profile = slope * self.LEVELS
        var = np.broadcast_to(profile[None, :, None, None],
                              (1, 3, 3, 3)).copy()
        omega_value = 0.5
        _, v = calculate_advection(
            var, self._uniform(0.0), self._uniform(0.0),
            self._uniform(omega_value), LATS, LONS, self.LEVELS)
        assert np.allclose(v, -omega_value * slope, rtol=1e-6)

    def test_output_is_shaped_time_by_level(self):
        var = self._uniform(300.0, shape=(4, 3, 3, 3))
        h, v = calculate_advection(
            var, self._uniform(1.0, (4, 3, 3, 3)),
            self._uniform(1.0, (4, 3, 3, 3)),
            self._uniform(0.0, (4, 3, 3, 3)),
            LATS, LONS, self.LEVELS)
        assert h.shape == (4, 3)
        assert v.shape == (4, 3)

    def test_rejects_a_pole_centred_stencil(self):
        # The metric factor cos(phi) collapses at the pole. _stencil_indices
        # clamps a requested lat of 90 down to 89.75, so this guard is the
        # backstop for a stencil assembled by hand.
        polar = np.array([90.25, 90.0, 89.75])
        with pytest.raises(ValueError, match="polar"):
            calculate_advection(
                self._uniform(300.0), self._uniform(1.0), self._uniform(1.0),
                self._uniform(0.0), polar, LONS, self.LEVELS)


class TestRadiativeHeating:
    LEVELS = np.array([10_000.0, 50_000.0, 100_000.0])

    def test_balanced_column_does_not_heat(self):
        out = calculate_radiative_heating(
            swnet_top=[200.0], swnet_sfc=[200.0],
            lwnet_top=[-100.0], lwnet_sfc=[-100.0],
            pressure=self.LEVELS)
        assert np.allclose(out, 0.0)

    def test_returns_time_by_level_and_is_uniform_in_the_vertical(self):
        out = calculate_radiative_heating(
            swnet_top=[200.0, 150.0], swnet_sfc=[180.0, 140.0],
            lwnet_top=[-240.0, -240.0], lwnet_sfc=[-60.0, -60.0],
            pressure=self.LEVELS)
        assert out.shape == (2, len(self.LEVELS))
        # It is a bulk column diagnostic: every level carries the same rate.
        assert np.allclose(out, out[:, :1])

    def test_matches_the_analytic_mass_weighted_rate(self):
        convergence = -100.0  # W m-2 net loss from the column
        out = calculate_radiative_heating(
            swnet_top=[0.0], swnet_sfc=[0.0],
            lwnet_top=[convergence], lwnet_sfc=[0.0],
            pressure=self.LEVELS)
        depth = self.LEVELS.max() - self.LEVELS.min()
        expected = _GRAVITY * convergence / (_CP_DRY * depth)
        assert out[0, 0] == pytest.approx(expected)

    def test_magnitude_is_a_few_kelvin_per_day_for_realistic_fluxes(self):
        # Typical clear-sky values: the atmosphere cools at order 1-2 K/day.
        out = calculate_radiative_heating(
            swnet_top=[300.0], swnet_sfc=[250.0],
            lwnet_top=[-240.0], lwnet_sfc=[-60.0],
            pressure=self.LEVELS)
        k_per_day = out[0, 0] * 86400.0
        assert -5.0 < k_per_day < 0.0

    def test_surface_pressure_sets_the_column_depth(self):
        kwargs = dict(swnet_top=[300.0], swnet_sfc=[250.0],
                      lwnet_top=[-240.0], lwnet_sfc=[-60.0],
                      pressure=self.LEVELS)
        shallow = calculate_radiative_heating(surface_pressure=[50_000.0],
                                              **kwargs)
        deep = calculate_radiative_heating(surface_pressure=[100_000.0],
                                           **kwargs)
        # The same flux convergence spread over less mass heats/cools faster.
        assert abs(shallow[0, 0]) > abs(deep[0, 0])


class TestFluxRateNormalisation:
    """Regression: accumulated J m-2 fluxes must become W m-2 rates.

    Files written by the current downloader are already rates; those from the
    older CDS workflow hold hourly accumulations. Feeding the latter straight
    into the heating calculation overstated it by a factor of 3600.
    """

    def _flux(self, units):
        return xr.DataArray(np.array([3600.0]), dims=["time"],
                            attrs={"units": units})

    @pytest.mark.parametrize("units", ["J m**-2", "J m-2", "J m^-2"])
    def test_accumulations_are_divided_by_the_accumulation_period(self, units):
        assert float(_as_flux_rate(self._flux(units))[0]) == pytest.approx(1.0)

    @pytest.mark.parametrize("units", ["W m**-2", "W m-2"])
    def test_rates_pass_through_unchanged(self, units):
        assert float(_as_flux_rate(self._flux(units))[0]) == pytest.approx(3600.0)

    def test_an_unlabelled_flux_is_left_alone(self):
        assert float(_as_flux_rate(self._flux(""))[0]) == pytest.approx(3600.0)


class TestPressureCoordinateConversion:
    """Regression for the hPa -> Pa coordinate bug.

    Rescaling the level coordinate without re-indexing left the original hPa
    coordinate attached, so xarray outer-joined 37 hPa levels with 37 Pa levels
    into 68 and the advection step failed on the shape mismatch.
    """

    def _synthetic_era5(self, level_units):
        nt, nz = 3, 5
        levels = np.array([200.0, 400.0, 600.0, 800.0, 1000.0])
        if level_units == "Pa":
            levels = levels * 100.0

        time = np.array([np.datetime64("2019-01-01T00") + np.timedelta64(h, "h")
                         for h in range(nt)])
        shape = (nt, nz, 3, 3)

        def field(value):
            return (("time", "levels", "latitude", "longitude"),
                    np.full(shape, float(value)))

        temperature = np.broadcast_to(
            np.linspace(220.0, 290.0, nz)[None, :, None, None], shape).copy()
        geopotential = np.broadcast_to(
            np.linspace(11e4, 1e3, nz)[None, :, None, None], shape).copy()

        return xr.Dataset(
            {
                "t": (("time", "levels", "latitude", "longitude"), temperature),
                "z": (("time", "levels", "latitude", "longitude"), geopotential),
                "q": field(5e-3),
                "u": field(6.0),
                "v": field(-2.0),
                "w": field(0.05),
                "sp": (("time", "latitude", "longitude"),
                       np.full((nt, 3, 3), 95_000.0)),
                "t2m": (("time", "latitude", "longitude"),
                        np.full((nt, 3, 3), 288.0)),
            },
            coords={"time": time, "levels": levels,
                    "latitude": LATS, "longitude": LONS},
        )

    def test_hpa_levels_are_promoted_to_pa_without_duplication(self):
        out = era5_to_scm_forcing(self._synthetic_era5("hPa"))
        assert out.sizes["levels"] == 5, "levels were outer-joined, not converted"
        assert out.levels.values.max() == pytest.approx(100_000.0)
        assert out.levels.attrs["units"] == "Pa"

    def test_levels_already_in_pa_are_left_alone(self):
        out = era5_to_scm_forcing(self._synthetic_era5("Pa"))
        assert out.sizes["levels"] == 5
        assert out.levels.values.max() == pytest.approx(100_000.0)

    def test_both_unit_conventions_give_identical_forcings(self):
        from_hpa = era5_to_scm_forcing(self._synthetic_era5("hPa"))
        from_pa = era5_to_scm_forcing(self._synthetic_era5("Pa"))
        for name in ("h_advec_thetail", "v_advec_qt", "u_g", "w_ls"):
            np.testing.assert_allclose(
                from_hpa[name].values, from_pa[name].values,
                rtol=1e-9, atol=0.0,
                err_msg=f"{name} differs between hPa and Pa inputs",
            )

    def test_no_forcing_field_comes_out_all_nan(self):
        # An outer join fills the mismatched half with NaN, so this catches the
        # same bug from the other side.
        out = era5_to_scm_forcing(self._synthetic_era5("hPa"))
        for name, da in out.data_vars.items():
            assert not np.all(np.isnan(da.values)), f"{name} is entirely NaN"
