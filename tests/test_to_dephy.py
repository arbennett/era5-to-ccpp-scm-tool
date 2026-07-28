"""DEPHY output: schema, metadata, and the companion namelist."""

import json
from datetime import datetime

import numpy as np
import pytest
import xarray as xr

from era5_to_ccpp_scm.to_dephy import (
    _NSOIL,
    _interp_ozone,
    _parse_date,
    convert_to_dephy,
    override_land_state,
    write_case_namelist,
)

#: Every variable a DEPHY land+LSM driver must carry for CCPP-SCM v7.0.0.
#: Checked against the reference case ``gabls3_noahmp_SCM_driver.nc``; a
#: missing name here is what a silently unrunnable case looks like.
EXPECTED_DEPHY_VARIABLES = {
    # initial state at t0
    "thetal", "qt", "ua", "va", "pa", "zh", "ps", "ql", "qi", "tke", "o3",
    "stc", "smc", "slc",
    # time-varying forcing
    "ps_forc", "pa_forc", "zh_forc", "tnthetal_adv", "tnqt_adv", "wap",
    "ug", "vg",
    # geometry
    "lat", "lon", "area", "soil_depth",
    # land-surface scalars
    "slmsk", "vegtyp", "soiltyp", "slopetyp", "tsfco", "tsfcl", "vegfrac",
    "shdmin", "shdmax", "canopy", "hice", "fice", "tisfc", "snowd", "snoalb",
    "tg3", "uustar", "weasd", "sncovr",
    "alvsf", "alnsf", "alvwf", "alnwf", "facsf", "facwf",
    "zorl", "zorll", "zorli", "zorlw",
}


@pytest.fixture(scope="module")
def dephy(tmp_path_factory, walnut_forcings):
    """A DEPHY driver built from the committed Walnut Gulch extraction."""
    out = tmp_path_factory.mktemp("dephy") / "fluxnet_US-Whs_SCM_driver.nc"
    convert_to_dephy(
        era5_processed=walnut_forcings,
        start_date="2019-01-01",
        output_file=str(out),
        case_name="fluxnet_US-Whs",
    )
    # decode_times=False keeps the CF time axis as raw seconds-since values,
    # which is the form the SCM reads and the form these tests assert on.
    with xr.open_dataset(str(out), decode_times=False) as ds:
        yield ds.load()


class TestParseDate:
    def test_accepts_a_bare_date(self):
        assert _parse_date("2019-01-01") == datetime(2019, 1, 1)

    def test_accepts_a_full_timestamp(self):
        assert _parse_date("2019-01-01 06:30:00") == \
            datetime(2019, 1, 1, 6, 30)

    def test_rejects_anything_else(self):
        with pytest.raises(ValueError, match="Cannot parse date"):
            _parse_date("2019/01/01")


class TestDephySchema:
    def test_carries_every_required_variable(self, dephy):
        missing = EXPECTED_DEPHY_VARIABLES - set(dephy.data_vars)
        assert not missing, f"DEPHY driver is missing {sorted(missing)}"

    def test_snow_depth_is_present_under_its_dephy_name(self, dephy):
        # Regression: the legacy template spells this 'snwdph', so without an
        # alias map 'snowd' silently vanished from the driver.
        assert "snowd" in dephy.data_vars
        assert np.isfinite(dephy["snowd"].values).all()

    def test_has_the_expected_dimensions(self, dephy):
        assert dephy.sizes["t0"] == 1
        assert dephy.sizes["nsoil"] == _NSOIL
        assert dephy.sizes["lev"] == 37
        assert dephy.sizes["time"] == 48

    def test_initial_condition_fields_are_shaped_t0_by_lev(self, dephy):
        for name in ("thetal", "qt", "ua", "va", "pa", "zh", "ql", "qi",
                     "tke", "o3"):
            assert dephy[name].dims == ("t0", "lev"), name

    def test_forcing_fields_are_shaped_time_by_lev(self, dephy):
        for name in ("pa_forc", "zh_forc", "tnthetal_adv", "tnqt_adv",
                     "wap", "ug", "vg"):
            assert dephy[name].dims == ("time", "lev"), name

    def test_soil_fields_are_shaped_t0_by_nsoil(self, dephy):
        for name in ("stc", "smc", "slc"):
            assert dephy[name].dims == ("t0", "nsoil"), name

    def test_everything_is_written_in_double_precision(self, dephy):
        for name, da in dephy.data_vars.items():
            if da.dtype.kind == "f":
                assert da.dtype == np.float64, name

    def test_no_field_contains_a_nan(self, dephy):
        for name, da in dephy.data_vars.items():
            if da.dtype.kind == "f":
                assert np.isfinite(da.values).all(), f"{name} has non-finite values"


class TestDephyTimeAxis:
    def test_time_is_seconds_since_the_start_date(self, dephy):
        assert dephy["time"].attrs["units"] == "seconds since 2019-01-01 00:00:00"

    def test_time_starts_at_zero_and_increases(self, dephy):
        times = dephy["time"].values
        assert times[0] == pytest.approx(0.0)
        assert np.all(np.diff(times) > 0)

    def test_hourly_era5_gives_an_hourly_forcing_axis(self, dephy):
        assert np.allclose(np.diff(dephy["time"].values), 3600.0)

    def test_t0_is_the_simulation_start(self, dephy):
        assert dephy["t0"].values[0] == pytest.approx(0.0)


class TestDephyPhysicalContent:
    def test_pressure_levels_are_in_pascals(self, dephy):
        levels = dephy["lev"].values
        assert levels.min() >= 100.0
        assert levels.max() == pytest.approx(100_000.0)

    def test_advective_tendency_is_the_sum_of_its_parts(self, walnut_forcings,
                                                        dephy):
        expected = (walnut_forcings["h_advec_thetail"].values.T
                    + walnut_forcings["v_advec_thetail"].values.T)
        np.testing.assert_allclose(dephy["tnthetal_adv"].values, expected,
                                   rtol=1e-12)

    def test_initial_state_is_the_first_forcing_time(self, walnut_forcings,
                                                     dephy):
        np.testing.assert_allclose(
            dephy["thetal"].values[0],
            walnut_forcings["thil_nudge"].isel(time=0).values, rtol=1e-12)

    def test_skin_temperature_is_taken_from_era5_not_the_template(
            self, walnut_forcings, dephy):
        # tsfcl and tsfco are skin temperatures, so they come from ERA5's skt
        # rather than from the 2 m air temperature that stands in only when no
        # land state could be derived.
        skin = float(walnut_forcings["tsfcl"].values)
        assert float(dephy["tsfcl"].values) == pytest.approx(skin, abs=1e-6)
        assert float(dephy["tsfco"].values) == pytest.approx(skin, abs=1e-6)
        assert skin != pytest.approx(float(walnut_forcings["T_surf"].values[0]))

    def test_site_coordinates_match_the_extraction_centre(self, walnut_forcings,
                                                          dephy):
        assert np.allclose(dephy["lat"].values,
                           float(walnut_forcings["latitude"].values))
        assert np.allclose(dephy["lon"].values,
                           float(walnut_forcings["longitude"].values))

    def test_ozone_is_finite_and_non_negative(self, dephy):
        # The GABLS3 template carries an all-zero ozone profile, so this is a
        # shape-and-sign check rather than a physical one. See
        # TestOzoneInterpolation for the interpolation itself.
        o3 = dephy["o3"].values
        assert o3.shape == (1, dephy.sizes["lev"])
        assert np.isfinite(o3).all()
        assert np.all(o3 >= 0.0)

    def test_soil_layers_are_the_four_noah_depths(self, dephy):
        np.testing.assert_allclose(dephy["soil_depth"].values,
                                   [0.1, 0.4, 1.0, 2.0])

    def test_cloud_water_and_tke_start_at_zero(self, dephy):
        for name in ("ql", "qi", "tke"):
            assert np.allclose(dephy[name].values, 0.0), name


class TestDephyGlobalAttributes:
    def test_declares_the_dephy_format_version(self, dephy):
        assert dephy.attrs["format_version"] == "DEPHY SCM format version 1"

    def test_advects_thetal_and_qt_only(self, dephy):
        assert dephy.attrs["adv_thetal"] == 1
        assert dephy.attrs["adv_qt"] == 1
        for off in ("adv_ta", "adv_qv", "adv_ua", "adv_va", "adv_theta"):
            assert dephy.attrs[off] == 0, off

    def test_uses_pressure_velocity_and_geostrophic_forcing(self, dephy):
        assert dephy.attrs["forc_wap"] == 1
        assert dephy.attrs["forc_wa"] == 0
        assert dephy.attrs["forc_geo"] == 1

    def test_all_nudging_is_disabled(self, dephy):
        for key, value in dephy.attrs.items():
            if key.startswith("nudging_"):
                assert value == 0, key

    def test_radiation_is_computed_by_the_scm(self, dephy):
        # Consequently the diagnosed dT_dt_rad is not used by the run.
        assert dephy.attrs["radiation"] == "on"

    def test_configured_as_a_land_case_driving_an_lsm(self, dephy):
        assert dephy.attrs["surface_type"] == "land"
        assert dephy.attrs["surface_forcing_lsm"] == "lsm"

    def test_records_the_simulation_window(self, dephy):
        assert dephy.attrs["start_date"] == "2019-01-01 00:00:00"
        assert dephy.attrs["end_date"] == "2019-01-02 23:00:00"


class TestScalarOverrides:
    def test_roughness_length_override_reaches_the_file(self, tmp_path,
                                                        walnut_forcings):
        out = tmp_path / "override_SCM_driver.nc"
        convert_to_dephy(
            era5_processed=walnut_forcings,
            start_date="2019-01-01",
            output_file=str(out),
            case_name="override",
            scalars_override={"zorl": 42.0, "zorll": 42.0},
        )
        with xr.open_dataset(str(out)) as ds:
            assert float(ds["zorl"].values) == pytest.approx(42.0)
            assert float(ds["zorll"].values) == pytest.approx(42.0)

    def test_unknown_override_names_are_ignored(self, tmp_path,
                                                walnut_forcings):
        out = tmp_path / "unknown_SCM_driver.nc"
        convert_to_dephy(
            era5_processed=walnut_forcings,
            start_date="2019-01-01",
            output_file=str(out),
            case_name="unknown",
            scalars_override={"not_a_dephy_variable": 1.0},
        )
        with xr.open_dataset(str(out)) as ds:
            assert "not_a_dephy_variable" not in ds.data_vars


class TestOzoneInterpolation:
    """The template's ozone profile is on its own levels and must be mapped
    onto the ERA5 levels without regard to either array's sort order."""

    def _template(self, levels, ozone):
        index = xr.Dataset(coords={"levels": np.asarray(levels, dtype=float)})
        initial = xr.Dataset(
            {"ozone": ("levels", np.asarray(ozone, dtype=float))},
            coords={"levels": np.asarray(levels, dtype=float)},
        )
        return index, initial

    def test_reproduces_the_template_on_matching_levels(self):
        levels = [10_000.0, 50_000.0, 100_000.0]
        ozone = [8e-6, 2e-7, 5e-8]
        index, initial = self._template(levels, ozone)
        out = _interp_ozone(index, initial, np.array(levels))
        np.testing.assert_allclose(out[0], ozone)

    def test_interpolates_linearly_between_template_levels(self):
        index, initial = self._template([0.0, 100_000.0], [0.0, 1.0])
        out = _interp_ozone(index, initial, np.array([25_000.0, 75_000.0]))
        np.testing.assert_allclose(out[0], [0.25, 0.75])

    def test_result_follows_the_order_of_the_target_levels(self):
        index, initial = self._template([0.0, 100_000.0], [0.0, 1.0])
        ascending = _interp_ozone(index, initial,
                                  np.array([25_000.0, 75_000.0]))
        descending = _interp_ozone(index, initial,
                                   np.array([75_000.0, 25_000.0]))
        np.testing.assert_allclose(descending[0], ascending[0][::-1])

    def test_a_template_without_ozone_yields_zeros(self):
        index = xr.Dataset(coords={"levels": np.array([0.0, 100_000.0])})
        initial = xr.Dataset()
        out = _interp_ozone(index, initial, np.array([1.0, 2.0, 3.0]))
        assert out.shape == (1, 3)
        assert np.allclose(out, 0.0)


class TestCaseNamelist:
    def test_is_valid_fortran_namelist_the_scm_can_read(self, tmp_path):
        f90nml = pytest.importorskip("f90nml")
        path = tmp_path / "fluxnet_US-Whs.nml"
        write_case_namelist(case_name="fluxnet_US-Whs",
                            output_file=str(path),
                            sfc_roughness_length_cm=15.0,
                            column_area=145_000_000.0)

        nml = f90nml.read(str(path))["case_config"]
        assert nml["case_name"] == "fluxnet_US-Whs"
        assert nml["column_area"] == pytest.approx(145_000_000.0)
        assert nml["sfc_roughness_length_cm"] == pytest.approx(15.0)
        # A DEPHY case with land-surface initial conditions.
        assert nml["input_type"] == 1
        assert nml["lsm_ics"] is True

    def test_spinup_can_be_turned_off(self, tmp_path):
        f90nml = pytest.importorskip("f90nml")
        path = tmp_path / "nospinup.nml"
        write_case_namelist(case_name="nospinup", output_file=str(path),
                            do_spinup=False)
        assert f90nml.read(str(path))["case_config"]["do_spinup"] is False


class TestLandStateInDephy:
    """The DEPHY writer prefers ERA5 land state and falls back per field."""

    def _forcings_with_land(self, walnut_forcings):
        """Attach a synthetic land state to the committed extraction."""
        ds = walnut_forcings.copy()
        ds["stc"] = xr.DataArray(
            np.array([278.0, 279.0, 281.0, 284.0]), dims=("nsoil",))
        ds["smc"] = xr.DataArray(
            np.array([0.11, 0.13, 0.15, 0.17]), dims=("nsoil",))
        ds["slc"] = xr.DataArray(
            np.array([0.11, 0.13, 0.15, 0.17]), dims=("nsoil",))
        ds["vegtyp"] = xr.DataArray(7.0)
        ds["soiltyp"] = xr.DataArray(3.0)
        ds["vegfrac"] = xr.DataArray(0.24)
        ds.attrs["land_state_source"] = json.dumps(
            {"stc": "ERA5 stl1-4", "vegtyp": "ERA5 mapped to IGBP"})
        return ds

    def _build(self, tmp_path, forcings, **kwargs):
        out = tmp_path / "case_SCM_driver.nc"
        convert_to_dephy(era5_processed=forcings, start_date="2019-01-01",
                         output_file=str(out), case_name="case", **kwargs)
        return out

    def test_era5_soil_state_supersedes_the_template(self, tmp_path,
                                                     walnut_forcings):
        out = self._build(tmp_path, self._forcings_with_land(walnut_forcings))
        with xr.open_dataset(str(out)) as ds:
            np.testing.assert_allclose(ds["stc"].values[0],
                                       [278.0, 279.0, 281.0, 284.0])
            np.testing.assert_allclose(ds["smc"].values[0],
                                       [0.11, 0.13, 0.15, 0.17])
            assert float(ds["vegtyp"]) == 7.0
            assert float(ds["soiltyp"]) == 3.0

    def test_template_soil_is_used_when_era5_has_none(self, dephy):
        # The committed extraction predates the land group, so its soil
        # profile still comes from GABLS3 and says so.
        provenance = json.loads(dephy.attrs["land_state_source"])
        assert "template" in provenance["stc"]
        assert np.all(np.isfinite(dephy["stc"].values))

    def test_caller_override_beats_era5(self, tmp_path, walnut_forcings):
        out = self._build(tmp_path, self._forcings_with_land(walnut_forcings),
                          scalars_override={"vegfrac": 0.9})
        with xr.open_dataset(str(out)) as ds:
            assert float(ds["vegfrac"]) == pytest.approx(0.9)
            assert json.loads(
                ds.attrs["land_state_source"])["vegfrac"] == "user override"

    def test_provenance_covers_every_land_scalar(self, dephy):
        provenance = json.loads(dephy.attrs["land_state_source"])
        for name in ("stc", "smc", "slc", "vegtyp", "soiltyp", "tsfcl",
                     "vegfrac", "snoalb", "tg3"):
            assert name in provenance, f"{name} has no recorded provenance"

    def test_soil_moisture_units_are_volumetric(self, dephy):
        # The template group labels these m3 m-3 and the SCM reads them that
        # way; an earlier version of the writer said "kg".
        assert dephy["smc"].attrs["units"] == "m3 m-3"
        assert dephy["slc"].attrs["units"] == "m3 m-3"


class TestOverrideLandState:
    """Patching a finished case, which is how a site descriptor gets applied."""

    @pytest.fixture
    def driver(self, tmp_path, walnut_forcings):
        out = tmp_path / "fluxnet_US-Whs_SCM_driver.nc"
        convert_to_dephy(era5_processed=walnut_forcings,
                         start_date="2019-01-01", output_file=str(out),
                         case_name="fluxnet_US-Whs")
        return out

    def test_fluxnet_code_sets_the_vegetation_type(self, driver):
        applied = override_land_state(str(driver), vegtyp="OSH")
        assert applied["vegtyp"] == 7
        with xr.open_dataset(str(driver)) as ds:
            assert float(ds["vegtyp"]) == 7.0

    def test_soil_type_accepts_a_class_name(self, driver):
        override_land_state(str(driver), soiltyp="sandy loam")
        with xr.open_dataset(str(driver)) as ds:
            assert float(ds["soiltyp"]) == 3.0

    def test_writing_elsewhere_leaves_the_original_alone(self, driver,
                                                         tmp_path):
        with xr.open_dataset(str(driver)) as ds:
            before = float(ds["vegtyp"])

        patched = tmp_path / "patched.nc"
        override_land_state(str(driver), str(patched), vegtyp="DBF")

        with xr.open_dataset(str(driver)) as ds:
            assert float(ds["vegtyp"]) == before
        with xr.open_dataset(str(patched)) as ds:
            assert float(ds["vegtyp"]) == 4.0

    def test_override_is_recorded_in_provenance_and_modifications(self, driver):
        override_land_state(str(driver), vegtyp="GRA", vegfrac=0.35)
        with xr.open_dataset(str(driver)) as ds:
            provenance = json.loads(ds.attrs["land_state_source"])
            assert provenance["vegtyp"] == "user override"
            assert provenance["vegfrac"] == "user override"
            assert "vegtyp=10" in ds.attrs["modifications"]

    def test_the_rest_of_the_case_is_untouched(self, driver):
        with xr.open_dataset(str(driver)) as ds:
            before = ds["thetal"].values.copy()
        override_land_state(str(driver), vegtyp="GRA")
        with xr.open_dataset(str(driver)) as ds:
            np.testing.assert_allclose(ds["thetal"].values, before)

    def test_unknown_field_is_rejected(self, driver):
        with pytest.raises(ValueError, match="Cannot override"):
            override_land_state(str(driver), albedo=0.2)

    def test_no_overrides_is_rejected(self, driver):
        with pytest.raises(ValueError, match="No overrides"):
            override_land_state(str(driver))
