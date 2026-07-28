"""Tests for the ERA5 to Noah land surface translation.

These are all network-free.  The numerical tests exercise the remapping and
parameter transfers directly; the integration tests build datasets in memory
that stand in for an ERA5 extraction.
"""

import numpy as np
import pytest
import xarray as xr

from era5_to_ccpp_scm.land_state import (
    ECMWF_TO_IGBP,
    ECMWF_TO_STATSGO,
    ERA5_SOIL_INTERFACES,
    FLUXNET_IGBP_CODES,
    IGBP_CLASS_NAMES,
    NOAH_SOIL_INTERFACES,
    STATSGO_CLASS_NAMES,
    derive_land_state,
    htessel_parameters,
    liquid_soil_moisture,
    map_soil_type,
    map_vegetation_type,
    parse_soil_type,
    parse_vegetation_type,
    remap_soil_layers,
    rescale_soil_moisture,
    snow_cover_fraction,
    soil_layer_weights,
    statsgo_parameters,
)


class TestSoilLayerRemap:
    def test_weights_form_a_weighted_mean(self):
        weights = soil_layer_weights()
        assert weights.shape == (4, 4)
        assert np.all(weights >= 0)
        np.testing.assert_allclose(weights.sum(axis=1), 1.0, rtol=1e-12)

    def test_constant_profile_is_unchanged(self):
        remapped = remap_soil_layers([285.0, 285.0, 285.0, 285.0])
        np.testing.assert_allclose(remapped, 285.0, rtol=1e-12)

    def test_depth_integral_is_conserved(self):
        # The remap is an overlap-weighted mean, so the depth integral over the
        # span the two grids share must survive it exactly.
        source = np.array([0.30, 0.25, 0.18, 0.12])

        src_thickness = np.diff(ERA5_SOIL_INTERFACES)
        overlap = np.minimum(ERA5_SOIL_INTERFACES[1:],
                             NOAH_SOIL_INTERFACES[-1]) \
            - np.minimum(ERA5_SOIL_INTERFACES[:-1], NOAH_SOIL_INTERFACES[-1])
        expected = float((source * overlap).sum())
        assert src_thickness.shape == source.shape

        remapped = remap_soil_layers(source)
        got = float((remapped * np.diff(NOAH_SOIL_INTERFACES)).sum())
        assert got == pytest.approx(expected, rel=1e-12)

    def test_top_noah_layer_mixes_the_top_two_era5_layers(self):
        # Noah's 0-10 cm layer spans all of ERA5's 0-7 cm and part of its
        # 7-28 cm layer, in a 7:3 ratio.
        weights = soil_layer_weights()
        assert weights[0, 0] == pytest.approx(0.7, rel=1e-12)
        assert weights[0, 1] == pytest.approx(0.3, rel=1e-12)
        assert weights[0, 2:].sum() == 0.0

    def test_wrong_number_of_layers_is_rejected(self):
        with pytest.raises(ValueError, match="source layers"):
            remap_soil_layers([1.0, 2.0, 3.0])

    def test_non_monotonic_interfaces_are_rejected(self):
        with pytest.raises(ValueError, match="monotonically"):
            soil_layer_weights([0.0, 0.5, 0.2], [0.0, 1.0])


class TestSoilParameters:
    @pytest.mark.parametrize("soiltyp", range(1, 20))
    def test_moisture_thresholds_are_ordered(self, soiltyp):
        params = statsgo_parameters(soiltyp)
        assert 0 < params["theta_wilt"] < params["theta_ref"] < params["theta_sat"]

    def test_loam_matches_the_noah_table(self):
        # Class 6 is loam; redprm derives a wilting point near the 0.066 that
        # set_soilveg.f carries as its reference value.
        params = statsgo_parameters(6)
        assert params["theta_sat"] == pytest.approx(0.439)
        assert params["theta_wilt"] == pytest.approx(0.066, abs=0.005)

    @pytest.mark.parametrize("bad", [0, 20, -1])
    def test_out_of_range_class_is_rejected(self, bad):
        with pytest.raises(ValueError, match="soil type"):
            statsgo_parameters(bad)

    @pytest.mark.parametrize("slt", range(1, 8))
    def test_htessel_thresholds_are_ordered(self, slt):
        params = htessel_parameters(slt)
        assert 0 < params["theta_wilt"] < params["theta_sat"]


class TestSoilMoistureTransfer:
    def test_relative_mode_maps_the_endpoints(self):
        source = htessel_parameters(2)
        target = statsgo_parameters(6)

        at_saturation = rescale_soil_moisture(
            [source["theta_sat"]], era5_soiltyp=2, noah_soiltyp=6)
        at_wilting = rescale_soil_moisture(
            [source["theta_wilt"]], era5_soiltyp=2, noah_soiltyp=6)

        assert at_saturation[0] == pytest.approx(target["theta_sat"])
        assert at_wilting[0] == pytest.approx(target["theta_wilt"])

    def test_relative_mode_preserves_the_saturation_ratio(self):
        source = htessel_parameters(4)
        target = statsgo_parameters(9)
        halfway = 0.5 * (source["theta_sat"] + source["theta_wilt"])

        got = rescale_soil_moisture([halfway], 4, 9)[0]
        expected = 0.5 * (target["theta_sat"] + target["theta_wilt"])
        assert got == pytest.approx(expected)

    @pytest.mark.parametrize("mode", ["relative", "direct"])
    @pytest.mark.parametrize("noah_soiltyp", range(1, 20))
    def test_result_is_always_physical(self, mode, noah_soiltyp):
        target = statsgo_parameters(noah_soiltyp)
        # Sweep the full plausible ERA5 range, including values outside the
        # target soil's own bounds.
        theta = np.linspace(0.0, 0.8, 25)
        got = rescale_soil_moisture(theta, 5, noah_soiltyp, mode=mode)
        assert np.all(got >= target["theta_wilt"] - 1e-12)
        assert np.all(got <= target["theta_sat"] + 1e-12)

    def test_direct_mode_passes_an_in_range_value_through(self):
        # 0.30 sits inside loam's range, so a direct copy must not move it.
        assert rescale_soil_moisture([0.30], 2, 6, mode="direct")[0] == \
            pytest.approx(0.30)

    def test_unknown_mode_is_rejected(self):
        with pytest.raises(ValueError, match="transfer mode"):
            rescale_soil_moisture([0.3], 2, 6, mode="nearest")


class TestFrozenSoil:
    def test_all_water_is_liquid_above_freezing(self):
        smc = np.array([0.30, 0.28, 0.25, 0.22])
        stc = np.array([280.0, 285.0, 288.0, 290.0])
        np.testing.assert_allclose(liquid_soil_moisture(smc, stc, 6), smc)

    def test_frozen_soil_holds_less_liquid(self):
        smc = np.full(4, 0.30)
        stc = np.array([260.0, 268.0, 272.0, 280.0])
        liquid = liquid_soil_moisture(smc, stc, 6)

        assert np.all(liquid <= smc + 1e-12)
        assert liquid[0] < smc[0]
        # Colder soil holds less supercooled water, and the unfrozen layer is
        # untouched.
        assert liquid[0] < liquid[1] < liquid[2]
        assert liquid[3] == pytest.approx(0.30)

    def test_liquid_water_is_never_negative(self):
        smc = np.full(4, 0.30)
        stc = np.array([220.0, 230.0, 240.0, 250.0])
        assert np.all(liquid_soil_moisture(smc, stc, 3) >= 0.0)


class TestSnowCover:
    def test_bare_ground_has_no_cover(self):
        assert snow_cover_fraction(0.0, 10) == 0.0

    def test_deep_snow_is_fully_covered(self):
        assert snow_cover_fraction(500.0, 10) == 1.0

    def test_partial_cover_is_bounded_and_monotone(self):
        depths = [1.0, 5.0, 10.0, 15.0]
        fractions = [snow_cover_fraction(d, 10) for d in depths]
        assert all(0.0 < f < 1.0 for f in fractions)
        assert fractions == sorted(fractions)

    def test_forest_needs_more_snow_than_grass_to_cover(self):
        # SNUP is 0.08 m for forest classes against 0.02 m for grassland.
        assert snow_cover_fraction(20.0, 1) < snow_cover_fraction(20.0, 10)


class TestClassificationTables:
    def test_every_era5_vegetation_class_maps_into_igbp(self):
        assert sorted(ECMWF_TO_IGBP) == list(range(1, 21))
        assert all(v in IGBP_CLASS_NAMES for v in ECMWF_TO_IGBP.values())

    def test_every_era5_soil_class_maps_into_statsgo(self):
        assert sorted(ECMWF_TO_STATSGO) == list(range(1, 8))
        assert all(v in STATSGO_CLASS_NAMES for v in ECMWF_TO_STATSGO.values())

    def test_fluxnet_codes_cover_the_igbp_table(self):
        assert sorted(FLUXNET_IGBP_CODES.values()) == list(range(1, 21))

    def test_dominant_cover_selects_the_vegetation_type(self):
        # Evergreen needleleaf (3) over short grass (2): high cover wins.
        assert map_vegetation_type(tvl=2, tvh=3, cvl=0.2, cvh=0.8) == 1
        # Low cover wins when it dominates.
        assert map_vegetation_type(tvl=2, tvh=3, cvl=0.9, cvh=0.1) == 10

    def test_unvegetated_cell_falls_back_to_barren(self):
        assert map_vegetation_type(tvl=0, tvh=0, cvl=0.0, cvh=0.0) == 16

    def test_ocean_soil_type_is_rejected(self):
        with pytest.raises(ValueError, match="no STATSGO equivalent"):
            map_soil_type(0)


class TestClassParsing:
    @pytest.mark.parametrize("value,expected", [
        ("GRA", 10), ("gra", 10), ("DBF", 4), ("ENF", 1),
        ("grasslands", 10), ("mixed forest", 5), (10, 10), ("10", 10),
    ])
    def test_vegetation_type_accepts_every_spelling(self, value, expected):
        assert parse_vegetation_type(value) == expected

    @pytest.mark.parametrize("value,expected", [
        ("sandy loam", 3), ("clay", 12), (6, 6), ("6", 6),
    ])
    def test_soil_type_accepts_names_and_numbers(self, value, expected):
        assert parse_soil_type(value) == expected

    def test_unknown_vegetation_code_is_rejected(self):
        with pytest.raises(ValueError, match="Unrecognised vegetation type"):
            parse_vegetation_type("SHRUB")

    @pytest.mark.parametrize("bad", [0, 21, "0"])
    def test_out_of_range_vegetation_type_is_rejected(self, bad):
        with pytest.raises(ValueError):
            parse_vegetation_type(bad)


def _fake_extraction(**overrides):
    """Build a minimal stand-in for an ERA5 surface extraction."""
    times = np.array(["2019-01-15T00", "2019-01-15T01"], dtype="datetime64[ns]")
    coords = {"time": times, "latitude": [32.0, 31.75, 31.5],
              "longitude": [-110.3, -110.05, -109.8]}

    def field(value):
        return xr.DataArray(np.full((2, 3, 3), value, dtype=float),
                            dims=("time", "latitude", "longitude"),
                            coords=coords)

    data = {
        "skt": field(283.0),
        "stl1": field(281.0), "stl2": field(282.0),
        "stl3": field(284.0), "stl4": field(288.0),
        "swvl1": field(0.15), "swvl2": field(0.18),
        "swvl3": field(0.20), "swvl4": field(0.22),
        "sd": field(0.0), "rsn": field(150.0), "src": field(0.0002),
        "aluvp": field(0.12), "aluvd": field(0.14),
        "alnip": field(0.25), "alnid": field(0.27),
        "fsr": field(0.05), "lailv": field(1.0), "laihv": field(0.0),
    }
    data.update(overrides)
    return xr.Dataset(data)


def _fake_invariant(slt=2, tvl=2, tvh=0, cvl=0.6, cvh=0.0, lsm=1.0):
    coords = {"latitude": [32.0, 31.75, 31.5],
              "longitude": [-110.3, -110.05, -109.8]}

    def field(value):
        return xr.DataArray(np.full((3, 3), value, dtype=float),
                            dims=("latitude", "longitude"), coords=coords)

    return xr.Dataset({"slt": field(slt), "tvl": field(tvl), "tvh": field(tvh),
                       "cvl": field(cvl), "cvh": field(cvh), "lsm": field(lsm)})


class TestDeriveLandState:
    def test_full_extraction_yields_a_complete_state(self):
        state = derive_land_state(_fake_extraction(), _fake_invariant())

        assert state["vegtyp"] == 10          # short grass -> grasslands
        assert state["soiltyp"] == 6          # medium -> loam
        assert len(state["stc"]) == 4
        assert len(state["smc"]) == 4
        assert state["tsfcl"] == pytest.approx(283.0)
        assert state["tg3"] == pytest.approx(288.0)
        assert state["slmsk"] == 1.0
        assert state["zorl"] == pytest.approx(5.0)   # 0.05 m -> cm
        assert state["canopy"] == pytest.approx(0.2)  # 0.0002 m -> kg m-2
        assert state["alvsf"] == pytest.approx(0.12)

    def test_soil_temperature_is_remapped_not_copied(self):
        state = derive_land_state(_fake_extraction(), _fake_invariant())
        # Noah's top layer straddles ERA5's first two, so it cannot equal
        # either one, and must sit between them.
        assert 281.0 < state["stc"][0] < 282.0

    def test_empty_extraction_yields_nothing(self):
        assert derive_land_state(xr.Dataset()) is None

    def test_partial_extraction_degrades_gracefully(self):
        # A pre-land-group extraction carries a skin temperature and no soil.
        skin_only = _fake_extraction()[["skt"]]
        state = derive_land_state(skin_only)

        assert state["tsfcl"] == pytest.approx(283.0)
        assert "stc" not in state
        assert "vegtyp" not in state

    def test_missing_invariant_still_gives_a_soil_profile(self):
        state = derive_land_state(_fake_extraction(), invariant=None)
        assert "stc" in state and "smc" in state
        # Without the soil classes the two parameter tables cannot be
        # reconciled, and that is recorded rather than silently assumed.
        assert "soil classes unavailable" in state["provenance"]["smc"]

    def test_overrides_take_precedence_over_era5(self):
        state = derive_land_state(_fake_extraction(), _fake_invariant(),
                                  vegtyp="DBF", soiltyp="clay")
        assert state["vegtyp"] == 4
        assert state["soiltyp"] == 12
        assert state["provenance"]["vegtyp"] == "user override"

    def test_soil_class_changes_the_moisture_transfer(self):
        loam = derive_land_state(_fake_extraction(), _fake_invariant(),
                                 soiltyp="loam")
        sand = derive_land_state(_fake_extraction(), _fake_invariant(),
                                 soiltyp="sand")
        assert not np.allclose(loam["smc"], sand["smc"])

    def test_vegetation_fraction_follows_leaf_area(self):
        bare = derive_land_state(
            _fake_extraction(lailv=_fake_extraction()["lailv"] * 0.0),
            _fake_invariant())
        leafy = derive_land_state(_fake_extraction(), _fake_invariant())
        assert bare["vegfrac"] == pytest.approx(0.0)
        assert 0.0 < leafy["vegfrac"] < 1.0

    def test_snow_state_is_derived_when_snow_is_present(self):
        snowy = _fake_extraction()
        snowy["sd"] = snowy["sd"] + 0.05     # 50 mm water equivalent
        state = derive_land_state(snowy, _fake_invariant())

        assert state["weasd"] == pytest.approx(50.0)
        assert state["snowd"] == pytest.approx(50.0 * 1000.0 / 150.0)
        assert state["sncovr"] == 1.0

    def test_ocean_point_is_flagged_in_the_mask(self):
        state = derive_land_state(_fake_extraction(),
                                  _fake_invariant(lsm=0.0))
        assert state["slmsk"] == 0.0

    def test_provenance_names_every_derived_field(self):
        state = derive_land_state(_fake_extraction(), _fake_invariant())
        provenance = state["provenance"]
        for key in state:
            if key != "provenance":
                assert key in provenance, f"{key} has no provenance"
