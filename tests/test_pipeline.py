"""End-to-end conversion and CLI wiring.

The full pipeline's first stage reads the ERA5 archive, so it is not covered
here; everything downstream of the download runs against the committed Walnut
Gulch extraction.
"""

import os

import numpy as np
import pytest
import xarray as xr
from click.testing import CliRunner

from era5_to_ccpp_scm.cli import _core_convert_forcings, cli

#: Fields ``convert_to_dephy`` reads off the processed forcing dataset.
REQUIRED_FORCING_VARIABLES = {
    "w_ls", "omega", "u_nudge", "v_nudge", "T_nudge", "thil_nudge", "qt_nudge",
    "u_g", "v_g", "h_advec_thetail", "v_advec_thetail", "h_advec_qt",
    "v_advec_qt", "dT_dt_rad", "zh", "p_surf", "T_surf",
}


class TestConvertForcings:
    def test_produces_every_field_the_dephy_writer_needs(self, walnut_forcings):
        missing = REQUIRED_FORCING_VARIABLES - set(walnut_forcings.data_vars)
        assert not missing, f"forcing output is missing {sorted(missing)}"

    def test_cds_style_dimension_names_are_renamed(self, walnut_forcings):
        # The committed files use valid_time/pressure_level.
        assert set(walnut_forcings.dims) == {"time", "levels"}

    def test_levels_are_converted_to_pascals(self, walnut_forcings):
        assert walnut_forcings.levels.values.max() == pytest.approx(100_000.0)
        assert walnut_forcings.levels.attrs["units"] == "Pa"

    def test_retains_the_full_era5_vertical_grid(self, walnut_forcings):
        # 37 standard pressure levels; an outer-join bug turned this into 68.
        assert walnut_forcings.sizes["levels"] == 37

    def test_collapses_the_stencil_to_its_centre_point(self, walnut_forcings):
        for name, da in walnut_forcings.data_vars.items():
            assert "latitude" not in da.dims, name
            assert "longitude" not in da.dims, name

    def test_records_the_site_coordinates(self, walnut_forcings):
        # The centre of the stencil, near Walnut Gulch (31.7438N, 110.0522W).
        # These files came from CDS, whose subsetting does not land exactly on
        # the native 0.25 degree grid, so this is a proximity check.
        assert float(walnut_forcings["latitude"].values) == \
            pytest.approx(31.74, abs=0.15)
        assert float(walnut_forcings["longitude"].values) % 360 == \
            pytest.approx(249.95, abs=0.15)

    def test_no_forcing_field_is_entirely_nan(self, walnut_forcings):
        for name, da in walnut_forcings.data_vars.items():
            assert not np.all(np.isnan(da.values)), f"{name} is entirely NaN"

    def test_variables_carry_units_and_a_long_name(self, walnut_forcings):
        for name in REQUIRED_FORCING_VARIABLES - {"latitude", "longitude"}:
            attrs = walnut_forcings[name].attrs
            assert "units" in attrs, name
            assert "long_name" in attrs, name


class TestForcingMagnitudes:
    """Order-of-magnitude checks. These are loose on purpose: they catch unit
    errors and sign flips without pinning the numbers to a fixed answer."""

    def test_surface_pressure_is_plausible_for_a_1200_m_site(self, walnut_forcings):
        p = walnut_forcings["p_surf"].values
        assert np.all((80_000.0 < p) & (p < 92_000.0))

    def test_surface_temperature_is_a_plausible_january_desert_range(
            self, walnut_forcings):
        t = walnut_forcings["T_surf"].values
        assert np.all((250.0 < t) & (t < 310.0))

    def test_geopotential_height_increases_as_pressure_falls(self,
                                                             walnut_forcings):
        zh = walnut_forcings["zh"].isel(time=0)
        levels = walnut_forcings.levels.values
        order = np.argsort(levels)
        assert np.all(np.diff(zh.values[order]) < 0)

    def test_vertical_velocity_stays_within_a_few_cm_per_second(self,
                                                                walnut_forcings):
        # Synoptic-scale w over land; metres per second, not Pa per second.
        assert np.nanmax(np.abs(walnut_forcings["w_ls"].values)) < 1.0

    def test_geostrophic_winds_are_of_synoptic_magnitude(self, walnut_forcings):
        for name in ("u_g", "v_g"):
            assert np.nanmax(np.abs(walnut_forcings[name].values)) < 150.0

    def test_radiative_heating_is_a_few_kelvin_per_day(self, walnut_forcings):
        # Regression: the legacy J m-2 rad file made this 3600x too large.
        k_per_day = walnut_forcings["dT_dt_rad"].values * 86400.0
        assert np.nanmax(np.abs(k_per_day)) < 20.0
        assert np.nanmean(k_per_day) < 0.0, "the column should cool on average"

    def test_specific_humidity_is_a_mass_fraction(self, walnut_forcings):
        q = walnut_forcings["qt_nudge"].values
        assert np.all((q >= 0.0) & (q < 0.05))


class TestRadiationIsOptional:
    def test_conversion_succeeds_without_a_radiation_file(self, walnut_files):
        sfc, pl, _ = walnut_files
        out = _core_convert_forcings(sfc, pl, None, None)
        # dT_dt_rad is zero-filled rather than absent, so the DEPHY writer and
        # the legacy template writer both still find it.
        assert "dT_dt_rad" in out
        assert np.allclose(out["dT_dt_rad"].values, 0.0)

    def test_the_other_forcings_are_unaffected(self, walnut_files,
                                               walnut_forcings):
        sfc, pl, _ = walnut_files
        out = _core_convert_forcings(sfc, pl, None, None)
        for name in ("h_advec_thetail", "u_g", "w_ls", "zh"):
            np.testing.assert_allclose(out[name].values,
                                       walnut_forcings[name].values,
                                       rtol=1e-12)


class TestCli:
    def test_help_lists_every_subcommand(self):
        result = CliRunner().invoke(cli, ["--help"])
        assert result.exit_code == 0
        for command in ("download_era5", "convert_forcings",
                        "convert_to_dephy", "convert_era5_from_template",
                        "run_full_pipeline"):
            assert command in result.output

    @pytest.mark.parametrize("command", [
        "download_era5", "convert_forcings", "convert_to_dephy",
        "convert_era5_from_template", "run_full_pipeline",
    ])
    def test_each_subcommand_has_help(self, command):
        result = CliRunner().invoke(cli, [command, "--help"])
        assert result.exit_code == 0

    def test_convert_forcings_writes_a_readable_file(self, tmp_path,
                                                     walnut_files):
        sfc, pl, rad = walnut_files
        out = tmp_path / "forcings.nc"
        result = CliRunner().invoke(cli, [
            "convert_forcings", "-s", sfc, "-p", pl, "-r", rad,
            "-o", str(out),
        ])
        assert result.exit_code == 0, result.output
        with xr.open_dataset(str(out)) as ds:
            assert REQUIRED_FORCING_VARIABLES <= set(ds.data_vars)

    def test_convert_to_dephy_writes_driver_and_namelist(self, tmp_path,
                                                         walnut_files):
        sfc, pl, rad = walnut_files
        forcings = tmp_path / "forcings.nc"
        _core_convert_forcings(sfc, pl, str(forcings), rad)

        driver = tmp_path / "fluxnet_US-Whs_SCM_driver.nc"
        result = CliRunner().invoke(cli, [
            "convert_to_dephy",
            "-f", str(forcings),
            "--start_date", "2019-01-01",
            "--case_name", "fluxnet_US-Whs",
            "-o", str(driver),
        ])
        assert result.exit_code == 0, result.output
        assert driver.exists()
        # The namelist name is derived from the driver name.
        assert (tmp_path / "fluxnet_US-Whs.nml").exists()

    def test_convert_to_dephy_installs_into_an_scm_tree(self, tmp_path,
                                                        walnut_files):
        sfc, pl, rad = walnut_files
        forcings = tmp_path / "forcings.nc"
        _core_convert_forcings(sfc, pl, str(forcings), rad)

        cases_dir = tmp_path / "processed_case_input"
        config_dir = tmp_path / "case_config"
        cases_dir.mkdir()
        config_dir.mkdir()

        result = CliRunner().invoke(cli, [
            "convert_to_dephy",
            "-f", str(forcings),
            "--start_date", "2019-01-01",
            "--case_name", "fluxnet_US-Whs",
            "-o", str(tmp_path / "fluxnet_US-Whs_SCM_driver.nc"),
            "--scm_cases_dir", str(cases_dir),
            "--scm_config_dir", str(config_dir),
        ])
        assert result.exit_code == 0, result.output
        # The SCM looks these up by exactly these names.
        assert os.path.exists(cases_dir / "fluxnet_US-Whs_SCM_driver.nc")
        assert os.path.exists(config_dir / "fluxnet_US-Whs.nml")


class TestLegacyTemplateOutput:
    def test_writes_the_grouped_format_with_all_four_groups(self, tmp_path,
                                                            walnut_files):
        sfc, pl, rad = walnut_files
        forcings = tmp_path / "forcings.nc"
        _core_convert_forcings(sfc, pl, str(forcings), rad)

        out = tmp_path / "legacy.nc"
        result = CliRunner().invoke(cli, [
            "convert_era5_from_template",
            "-f", str(forcings), "-t", "gabls3", "-o", str(out),
        ])
        assert result.exit_code == 0, result.output

        for group in (None, "forcing", "initial", "scalars"):
            with xr.open_dataset(str(out), group=group) as ds:
                assert len(ds.variables) > 0, f"group {group} is empty"

    def test_forcing_group_time_is_elapsed_seconds(self, tmp_path,
                                                   walnut_files):
        sfc, pl, rad = walnut_files
        forcings = tmp_path / "forcings.nc"
        _core_convert_forcings(sfc, pl, str(forcings), rad)

        out = tmp_path / "legacy.nc"
        CliRunner().invoke(cli, [
            "convert_era5_from_template",
            "-f", str(forcings), "-t", "gabls3", "-o", str(out),
        ])
        with xr.open_dataset(str(out), decode_times=False) as ds:
            times = ds["time"].values
        assert times[0] == pytest.approx(0.0)
        assert np.allclose(np.diff(times), 3600.0)
