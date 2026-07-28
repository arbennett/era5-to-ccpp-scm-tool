"""
Translate the ERA5 land surface state onto the Noah / Noah-MP grid and tables.

ERA5's land surface is HTESSEL, which differs from Noah in three ways that all
have to be reconciled before its state can be used as an initial condition:

  * **Soil layers.**  HTESSEL uses four layers with interfaces at 0, 7, 28, 100
    and 289 cm; Noah uses 0, 10, 40, 100 and 200 cm.  Fields are remapped by
    overlap weighting, which conserves the depth integral.
  * **Soil parameters.**  Porosity and wilting point differ between the two
    parameter tables, so a raw volumetric copy of soil moisture can land below
    the target soil type's wilting point and shut off transpiration entirely.
    The default transfer preserves relative wetness instead, because that ratio
    is precisely what Noah's water stress factor is computed from.
  * **Classification.**  HTESSEL land cover is a 20-class table derived from
    BATS and its soil texture is a 7-class table; the GFS physics used by the
    SCM is configured for IGBP (``ivegsrc = 1``) and STATSGO
    (``isot = 1``).  Both are mapped by lookup.

The Noah parameter values below are transcribed from ``set_soilveg.f`` in the
CCPP physics tree, which is the table the model itself uses.  Wilting point and
field capacity are recomputed here exactly as ``redprm`` recomputes them, since
the values printed in that file are marked as unused reference values.

A final caveat that no amount of care here removes: ERA5's land cover and soil
texture are 0.25 degree fields, roughly 28 km, which is a coarse descriptor of
an eddy covariance footprint.  Where site truth is known it should be supplied
instead, via the overrides in :func:`derive_land_state` or by patching a
finished case with :func:`era5_to_ccpp_scm.to_dephy.override_land_state`.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import xarray as xr


# ---------------------------------------------------------------------------
# Soil layer geometry
# ---------------------------------------------------------------------------

#: Depths of the HTESSEL layer interfaces, metres, top down.
ERA5_SOIL_INTERFACES = np.array([0.00, 0.07, 0.28, 1.00, 2.89])

#: Depths of the Noah / Noah-MP layer interfaces, metres, top down.
NOAH_SOIL_INTERFACES = np.array([0.00, 0.10, 0.40, 1.00, 2.00])

_NSOIL = 4


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

_T_FREEZE = 273.15        # K
_LATENT_FUSION = 3.335e5  # J kg-1
_GRAVITY = 9.80665        # m s-2

#: Tuning parameter in Noah's snow cover fraction, ``salp_data`` in set_soilveg.f
_SALP = 4.0

#: Extinction coefficient converting leaf area index to fractional canopy
#: closure through Beer's law, ``1 - exp(-k * LAI)``.  One half is the
#: conventional value for a randomly oriented canopy.
_LAI_EXTINCTION = 0.5

#: Soil moisture shape parameters used by ``redprm`` to derive the wilting
#: point and field capacity from the texture parameters.
_SMLOW = 0.5
_SMHIGH = 6.0


# ---------------------------------------------------------------------------
# Noah STATSGO soil parameters (isot = 1), classes 1-19
# ---------------------------------------------------------------------------

#: Clapp-Hornberger b exponent.
_STATSGO_B = np.array([
    4.05, 4.26, 4.74, 5.33, 5.33, 5.25, 6.77, 8.72, 8.17, 10.73,
    10.39, 11.55, 5.25, 4.26, 4.05, 4.26, 11.55, 4.05, 4.05,
])

#: Porosity, m3 m-3 (``MAXSMC``).
_STATSGO_THETA_SAT = np.array([
    0.395, 0.421, 0.434, 0.476, 0.476, 0.439, 0.404, 0.464, 0.465, 0.406,
    0.468, 0.457, 0.464, 0.421, 0.200, 0.421, 0.457, 0.200, 0.395,
])

#: Saturated matric potential, m (``SATPSI``).
_STATSGO_PSI_SAT = np.array([
    0.0350, 0.0363, 0.1413, 0.7586, 0.7586, 0.3548, 0.1349, 0.6166, 0.2630,
    0.0977, 0.3236, 0.4677, 0.3548, 0.0363, 0.0350, 0.0363, 0.4677, 0.0350,
    0.0350,
])

#: Saturated hydraulic conductivity, m s-1 (``SATDK``).
_STATSGO_K_SAT = np.array([
    1.7600e-4, 1.4078e-5, 5.2304e-6, 2.8089e-6, 2.8089e-6, 3.3770e-6,
    4.4518e-6, 2.0348e-6, 2.4464e-6, 7.2199e-6, 1.3444e-6, 9.7394e-7,
    3.3770e-6, 1.4078e-5, 1.4087e-5, 1.4078e-5, 9.7394e-7, 1.4078e-5,
    1.7600e-4,
])

#: Snow depth in metres of water equivalent above which snow cover is complete,
#: per IGBP vegetation class (``SNUPX``, ivegsrc = 1).
_IGBP_SNUP = np.array([
    0.080, 0.080, 0.080, 0.080, 0.080, 0.020, 0.020, 0.060, 0.040, 0.020,
    0.010, 0.020, 0.020, 0.020, 0.013, 0.013, 0.010, 0.020, 0.020, 0.020,
])


def statsgo_parameters(soiltyp: int) -> Dict[str, float]:
    """Return the Noah soil parameters for a STATSGO class.

    The wilting point and field capacity are derived the way ``redprm`` derives
    them rather than read from the table, because the tabulated values in
    ``set_soilveg.f`` are explicitly flagged as unused.

    Parameters
    ----------
    soiltyp : int
        STATSGO soil class, 1-19.

    Returns
    -------
    dict
        ``theta_sat``, ``theta_wilt``, ``theta_ref``, ``b`` and ``psi_sat``.
    """
    index = _validate_class(soiltyp, len(_STATSGO_THETA_SAT), "soil type")

    theta_sat = float(_STATSGO_THETA_SAT[index])
    b = float(_STATSGO_B[index])
    psi_sat = float(_STATSGO_PSI_SAT[index])
    k_sat = float(_STATSGO_K_SAT[index])

    wilt = theta_sat * (200.0 / psi_sat) ** (-1.0 / b)
    theta_wilt = wilt - _SMLOW * wilt

    ref = theta_sat * (5.79e-9 / k_sat) ** (1.0 / (2.0 * b + 3.0))
    theta_ref = ref + (theta_sat - ref) / _SMHIGH

    return {
        "theta_sat": theta_sat,
        "theta_wilt": theta_wilt,
        "theta_ref": theta_ref,
        "b": b,
        "psi_sat": psi_sat,
    }


# ---------------------------------------------------------------------------
# HTESSEL soil parameters, ERA5 soil types 1-7
# ---------------------------------------------------------------------------

#: Porosity and wilting point for the HTESSEL van Genuchten soil classes,
#: after Balsamo et al. (2009).  Class 7, tropical organic, is not separately
#: parameterised in the operational table and reuses the organic values.
_HTESSEL_THETA_SAT = np.array([0.403, 0.439, 0.430, 0.520, 0.614, 0.766, 0.766])
_HTESSEL_THETA_WILT = np.array([0.059, 0.151, 0.133, 0.279, 0.335, 0.267, 0.267])


def htessel_parameters(slt: int) -> Dict[str, float]:
    """Return the HTESSEL porosity and wilting point for an ERA5 soil type.

    Parameters
    ----------
    slt : int
        ERA5 soil type, 1-7.  Type 0 marks ocean and has no soil parameters.
    """
    index = _validate_class(slt, len(_HTESSEL_THETA_SAT), "ERA5 soil type")
    return {
        "theta_sat": float(_HTESSEL_THETA_SAT[index]),
        "theta_wilt": float(_HTESSEL_THETA_WILT[index]),
    }


def _validate_class(value, n_classes: int, label: str) -> int:
    """Convert a 1-based class index to a 0-based array index, or raise."""
    as_int = int(round(float(value)))
    if not 1 <= as_int <= n_classes:
        raise ValueError(
            f"{label} {as_int} is outside the valid range 1-{n_classes}."
        )
    return as_int - 1


# ---------------------------------------------------------------------------
# Classification lookups
# ---------------------------------------------------------------------------

#: ERA5 / HTESSEL vegetation type (1-20) to IGBP class (1-20).
#:
#: HTESSEL's table is derived from BATS and does not partition the land surface
#: the same way IGBP does, so several entries are judgement calls: the semidesert
#: and shrub classes in particular have no exact IGBP counterpart, and ERA5's
#: "interrupted forest" is mapped to woody savanna as the closest analogue in
#: canopy density.
ECMWF_TO_IGBP: Dict[int, int] = {
    1: 12,   # crops, mixed farming      -> croplands
    2: 10,   # short grass               -> grasslands
    3: 1,    # evergreen needleleaf      -> evergreen needleleaf forest
    4: 3,    # deciduous needleleaf      -> deciduous needleleaf forest
    5: 4,    # deciduous broadleaf       -> deciduous broadleaf forest
    6: 2,    # evergreen broadleaf       -> evergreen broadleaf forest
    7: 10,   # tall grass                -> grasslands
    8: 16,   # desert                    -> barren or sparsely vegetated
    9: 19,   # tundra                    -> mixed tundra
    10: 12,  # irrigated crops           -> croplands
    11: 7,   # semidesert                -> open shrublands
    12: 15,  # ice caps and glaciers     -> snow and ice
    13: 11,  # bogs and marshes          -> permanent wetlands
    14: 17,  # inland water              -> water
    15: 17,  # ocean                     -> water
    16: 6,   # evergreen shrubs          -> closed shrublands
    17: 7,   # deciduous shrubs          -> open shrublands
    18: 5,   # mixed forest              -> mixed forest
    19: 8,   # interrupted forest        -> woody savannas
    20: 14,  # water and land mixtures   -> cropland/natural vegetation mosaic
}

#: ERA5 soil type (1-7) to STATSGO class (1-19).
ECMWF_TO_STATSGO: Dict[int, int] = {
    1: 3,    # coarse       -> sandy loam
    2: 6,    # medium       -> loam
    3: 4,    # medium fine  -> silt loam
    4: 9,    # fine         -> clay loam
    5: 12,   # very fine    -> clay
    6: 13,   # organic      -> organic material
    7: 13,   # tropical organic -> organic material
}

#: IGBP class names, indexed by class number.
IGBP_CLASS_NAMES: Dict[int, str] = {
    1: "evergreen needleleaf forest",
    2: "evergreen broadleaf forest",
    3: "deciduous needleleaf forest",
    4: "deciduous broadleaf forest",
    5: "mixed forest",
    6: "closed shrublands",
    7: "open shrublands",
    8: "woody savannas",
    9: "savannas",
    10: "grasslands",
    11: "permanent wetlands",
    12: "croplands",
    13: "urban and built-up",
    14: "cropland/natural vegetation mosaic",
    15: "snow and ice",
    16: "barren or sparsely vegetated",
    17: "water",
    18: "wooded tundra",
    19: "mixed tundra",
    20: "bare ground tundra",
}

#: FLUXNET / AmeriFlux IGBP vegetation abbreviations to IGBP class number.
#: Site metadata distributes the class as one of these codes, so accepting them
#: directly is the least error-prone way to pin a case to its site descriptor.
FLUXNET_IGBP_CODES: Dict[str, int] = {
    "ENF": 1,   # evergreen needleleaf forest
    "EBF": 2,   # evergreen broadleaf forest
    "DNF": 3,   # deciduous needleleaf forest
    "DBF": 4,   # deciduous broadleaf forest
    "MF": 5,    # mixed forest
    "CSH": 6,   # closed shrublands
    "OSH": 7,   # open shrublands
    "WSA": 8,   # woody savannas
    "SAV": 9,   # savannas
    "GRA": 10,  # grasslands
    "WET": 11,  # permanent wetlands
    "CRO": 12,  # croplands
    "URB": 13,  # urban and built-up
    "CVM": 14,  # cropland/natural vegetation mosaic
    "SNO": 15,  # snow and ice
    "BSV": 16,  # barren or sparsely vegetated
    "WAT": 17,  # water
    "WTU": 18,  # wooded tundra
    "MTU": 19,  # mixed tundra
    "BTU": 20,  # bare ground tundra
}

#: STATSGO class names, indexed by class number.
STATSGO_CLASS_NAMES: Dict[int, str] = {
    1: "sand", 2: "loamy sand", 3: "sandy loam", 4: "silt loam", 5: "silt",
    6: "loam", 7: "sandy clay loam", 8: "silty clay loam", 9: "clay loam",
    10: "sandy clay", 11: "silty clay", 12: "clay", 13: "organic material",
    14: "water", 15: "bedrock", 16: "other", 17: "playa", 18: "lava",
    19: "white sand",
}


def parse_vegetation_type(value) -> int:
    """Resolve an IGBP class from a number, a name, or a FLUXNET code.

    Accepts an integer class, a FLUXNET/AmeriFlux abbreviation such as ``GRA``
    or ``DBF``, or a full IGBP class name such as ``"grasslands"``.  This is
    what lets a case be pinned to the descriptor published with a flux tower
    rather than to whatever ERA5 says at 0.25 degrees.
    """
    if isinstance(value, str):
        text = value.strip()
        if text.upper() in FLUXNET_IGBP_CODES:
            return FLUXNET_IGBP_CODES[text.upper()]
        for number, name in IGBP_CLASS_NAMES.items():
            if name == text.lower():
                return number
        try:
            value = int(text)
        except ValueError:
            raise ValueError(
                f"Unrecognised vegetation type {value!r}.  Give an IGBP class "
                f"number 1-20, an IGBP class name, or a FLUXNET code "
                f"({', '.join(sorted(FLUXNET_IGBP_CODES))})."
            ) from None
    _validate_class(value, 20, "vegetation type")
    return int(value)


def parse_soil_type(value) -> int:
    """Resolve a STATSGO class from a number or a class name."""
    if isinstance(value, str):
        text = value.strip()
        for number, name in STATSGO_CLASS_NAMES.items():
            if name == text.lower():
                return number
        try:
            value = int(text)
        except ValueError:
            raise ValueError(
                f"Unrecognised soil type {value!r}.  Give a STATSGO class "
                f"number 1-19 or a class name."
            ) from None
    _validate_class(value, 19, "soil type")
    return int(value)


def map_vegetation_type(tvl, tvh, cvl, cvh) -> int:
    """Choose an IGBP class from ERA5's low and high vegetation description.

    ERA5 describes a cell as a mixture of one low and one high vegetation type
    with independent cover fractions.  Noah carries a single class, so the type
    with the greater cover is taken, with high vegetation winning ties because
    a canopy dominates the surface energy balance where both are present.
    """
    low_type, high_type = int(round(float(tvl))), int(round(float(tvh)))
    low_cover, high_cover = float(cvl), float(cvh)

    if high_cover >= low_cover and high_type in ECMWF_TO_IGBP:
        return ECMWF_TO_IGBP[high_type]
    if low_type in ECMWF_TO_IGBP:
        return ECMWF_TO_IGBP[low_type]
    if high_type in ECMWF_TO_IGBP:
        return ECMWF_TO_IGBP[high_type]
    # Both types are 0, which ERA5 uses where there is no vegetation at all.
    return 16  # barren or sparsely vegetated


def map_soil_type(slt) -> int:
    """Map an ERA5 soil texture class onto a STATSGO class."""
    as_int = int(round(float(slt)))
    if as_int not in ECMWF_TO_STATSGO:
        raise ValueError(
            f"ERA5 soil type {as_int} has no STATSGO equivalent; valid ERA5 "
            f"types are {sorted(ECMWF_TO_STATSGO)} (0 marks ocean)."
        )
    return ECMWF_TO_STATSGO[as_int]


# ---------------------------------------------------------------------------
# Vertical remapping
# ---------------------------------------------------------------------------

def soil_layer_weights(src_interfaces=ERA5_SOIL_INTERFACES,
                       dst_interfaces=NOAH_SOIL_INTERFACES) -> np.ndarray:
    """Build the overlap-weight matrix that remaps one soil grid onto another.

    Entry ``(i, j)`` is the fraction of destination layer ``i`` that is covered
    by source layer ``j``, so applying the matrix takes a thickness-weighted
    mean over each destination layer.  This conserves the depth integral of the
    remapped quantity, which is what makes it correct for volumetric moisture
    and adequate for temperature.

    Returns
    -------
    numpy.ndarray
        Shape ``(len(dst_interfaces) - 1, len(src_interfaces) - 1)``, with rows
        summing to one wherever the destination layer is covered at all.
    """
    src = np.asarray(src_interfaces, dtype=float)
    dst = np.asarray(dst_interfaces, dtype=float)
    if np.any(np.diff(src) <= 0) or np.any(np.diff(dst) <= 0):
        raise ValueError("Soil interfaces must increase monotonically.")

    weights = np.zeros((len(dst) - 1, len(src) - 1))
    for i in range(len(dst) - 1):
        top, bottom = dst[i], dst[i + 1]
        for j in range(len(src) - 1):
            overlap = min(bottom, src[j + 1]) - max(top, src[j])
            if overlap > 0:
                weights[i, j] = overlap
        total = weights[i].sum()
        if total > 0:
            weights[i] /= total
    return weights


def remap_soil_layers(values, src_interfaces=ERA5_SOIL_INTERFACES,
                      dst_interfaces=NOAH_SOIL_INTERFACES) -> np.ndarray:
    """Remap a per-layer soil profile onto another set of layer interfaces."""
    values = np.asarray(values, dtype=float)
    weights = soil_layer_weights(src_interfaces, dst_interfaces)
    if values.shape[-1] != weights.shape[1]:
        raise ValueError(
            f"Expected {weights.shape[1]} source layers, got {values.shape[-1]}."
        )
    return weights @ values


# ---------------------------------------------------------------------------
# Soil moisture transfer
# ---------------------------------------------------------------------------

def rescale_soil_moisture(theta, era5_soiltyp: int, noah_soiltyp: int,
                          mode: str = "relative") -> np.ndarray:
    """Transfer volumetric soil moisture between two soil parameter tables.

    Two modes are available.

    ``relative`` (the default) preserves the degree of saturation between the
    wilting point and porosity::

        S      = (theta_ERA5 - wilt_HTESSEL) / (sat_HTESSEL - wilt_HTESSEL)
        theta  = wilt_STATSGO + S * (sat_STATSGO - wilt_STATSGO)

    Noah's water stress factor is this same ratio, so preserving it preserves
    the evaporative regime even though the water content changes.  A raw copy
    does not: HTESSEL porosities run well above STATSGO's for fine and organic
    soils, so copying a wet ERA5 clay into a Noah sandy loam can exceed
    porosity, while copying a dry ERA5 value into a finer Noah soil can land
    below wilting point and shut off transpiration entirely.

    ``direct`` copies the volumetric value and clamps it into the target soil's
    physical range.  It is the right choice when the two soil types already
    agree, or when comparing against a study that did the same.

    Returns
    -------
    numpy.ndarray
        Volumetric soil moisture on the Noah parameter set, clamped into
        ``[theta_wilt, theta_sat]``.
    """
    if mode not in ("relative", "direct"):
        raise ValueError(
            f"Unknown transfer mode {mode!r}; expected 'relative' or 'direct'."
        )

    theta = np.asarray(theta, dtype=float)
    target = statsgo_parameters(noah_soiltyp)

    if mode == "direct":
        rescaled = theta
    else:
        source = htessel_parameters(era5_soiltyp)
        span = source["theta_sat"] - source["theta_wilt"]
        saturation = (theta - source["theta_wilt"]) / span
        rescaled = target["theta_wilt"] + saturation * (
            target["theta_sat"] - target["theta_wilt"]
        )

    return np.clip(rescaled, target["theta_wilt"], target["theta_sat"])


def liquid_soil_moisture(smc, stc, noah_soiltyp: int) -> np.ndarray:
    """Split total soil moisture into its unfrozen part.

    ERA5 archives total soil water; Noah wants the liquid fraction separately.
    Above freezing all of it is liquid.  Below freezing the maximum liquid
    water that can coexist with ice follows from the freezing point depression
    implied by the Clapp-Hornberger retention curve,

        theta_liq_max = theta_sat * [ L_f (T_f - T) / (g psi_sat T) ] ^ (-1/b)

    which is the analytic solution Noah's ``frh2o`` uses as its starting guess.
    Using the target soil type's own ``b`` and ``psi_sat`` keeps the split
    consistent with the model that will read it.
    """
    smc = np.asarray(smc, dtype=float)
    stc = np.asarray(stc, dtype=float)
    params = statsgo_parameters(noah_soiltyp)

    frozen = stc < _T_FREEZE
    liquid = smc.copy()
    if np.any(frozen):
        temperature = stc[frozen]
        depression = (
            _LATENT_FUSION * (_T_FREEZE - temperature)
            / (_GRAVITY * params["psi_sat"] * temperature)
        )
        max_liquid = params["theta_sat"] * depression ** (-1.0 / params["b"])
        liquid[frozen] = np.minimum(smc[frozen], max_liquid)

    return liquid


def snow_cover_fraction(weasd_mm: float, vegtyp: int) -> float:
    """Diagnose fractional snow cover from snow water equivalent.

    ERA5 does not archive a snow cover fraction for the atmospheric model, so
    it is diagnosed with Noah's own ``snfrac`` relation and the IGBP-dependent
    depth threshold, which keeps the initial albedo consistent with what the
    model would compute from the same snow mass.

    Parameters
    ----------
    weasd_mm : float
        Snow water equivalent in mm, as stored in the DEPHY file.
    vegtyp : int
        IGBP vegetation class, which sets the threshold depth.
    """
    sneqv = float(weasd_mm) / 1000.0  # Noah works in metres of water
    if sneqv <= 0.0:
        return 0.0

    snup = float(_IGBP_SNUP[_validate_class(vegtyp, 20, "vegetation type")])
    if snup <= 0.0 or sneqv >= snup:
        return 1.0

    ratio = sneqv / snup
    return float(1.0 - (np.exp(-_SALP * ratio) - ratio * np.exp(-_SALP)))


# ---------------------------------------------------------------------------
# Assembling the land state
# ---------------------------------------------------------------------------

#: Soil profiles carried on the ``nsoil`` dimension.
SOIL_PROFILE_VARS = ("stc", "smc", "slc")

#: Scalar land surface fields this module can derive from ERA5.
LAND_SCALAR_VARS = (
    "tsfco", "tsfcl", "tisfc", "tg3", "weasd", "snowd", "sncovr", "snoalb",
    "canopy", "alvsf", "alvwf", "alnsf", "alnwf", "vegfrac", "shdmin",
    "shdmax", "vegtyp", "soiltyp", "slmsk", "zorl", "zorll",
)


class LandStateError(RuntimeError):
    """Raised when a land state is requested but cannot be derived."""


def derive_land_state(
    sfc: xr.Dataset,
    invariant: Optional[xr.Dataset] = None,
    time_index: int = 0,
    soil_moisture_transfer: str = "relative",
    vegtyp: Optional[object] = None,
    soiltyp: Optional[object] = None,
    center: int = 1,
) -> Optional[dict]:
    """Build a Noah-ready land surface state from an ERA5 extraction.

    Missing inputs degrade gracefully: whatever can be derived is returned and
    the rest is left for the caller to fill from its template.  ``None`` comes
    back only when the extraction carries no land fields at all, which is what
    happens for data downloaded before this feature existed.

    Parameters
    ----------
    sfc : xr.Dataset
        Surface extraction, carrying at least ``stl1``-``stl4`` and
        ``swvl1``-``swvl4`` for a soil state to be derivable.
    invariant : xr.Dataset, optional
        Time-invariant extraction carrying ``tvl``, ``tvh``, ``cvl``, ``cvh``
        and ``slt``.  Without it the vegetation and soil classes fall back to
        the caller's defaults, and the soil moisture transfer assumes the two
        tables agree.
    time_index : int
        Index into the time axis to take the state from; the default is the
        first time, which is the simulation start.
    soil_moisture_transfer : {'relative', 'direct'}
        See :func:`rescale_soil_moisture`.
    vegtyp, soiltyp : optional
        Explicit classes that override whatever ERA5 reports.  Accept a class
        number, a class name, or for vegetation a FLUXNET code.  Supplying the
        site's published descriptor here is preferable to ERA5's 0.25 degree
        land cover wherever it is known.
    center : int
        Index of the central point of the stencil in each horizontal dimension.

    Returns
    -------
    dict or None
        Keys are DEPHY variable names, plus ``provenance``, a mapping from each
        derived variable to a short description of where it came from.
    """
    if soil_moisture_transfer not in ("relative", "direct"):
        raise ValueError(
            f"Unknown transfer mode {soil_moisture_transfer!r}; "
            "expected 'relative' or 'direct'."
        )

    def _at_time(name):
        """Central-point value of a surface field at the chosen time."""
        if name not in sfc.variables:
            return None
        field = sfc[name]
        if "time" in field.dims:
            field = field.isel(time=time_index)
        return float(_center_value(field, center))

    def _invariant(name):
        if invariant is None or name not in invariant.variables:
            return None
        return float(_center_value(invariant[name], center))

    soil_temperature = [_at_time(f"stl{i}") for i in range(1, _NSOIL + 1)]
    soil_moisture = [_at_time(f"swvl{i}") for i in range(1, _NSOIL + 1)]
    has_soil = all(v is not None for v in soil_temperature + soil_moisture)

    state: Dict[str, object] = {}
    provenance: Dict[str, str] = {}

    # --- vegetation and soil class ------------------------------------
    era5_soiltyp = _invariant("slt")

    if soiltyp is not None:
        state["soiltyp"] = parse_soil_type(soiltyp)
        provenance["soiltyp"] = "user override"
    elif era5_soiltyp is not None and int(round(era5_soiltyp)) > 0:
        state["soiltyp"] = map_soil_type(era5_soiltyp)
        provenance["soiltyp"] = (
            f"ERA5 slt={int(round(era5_soiltyp))} mapped to STATSGO"
        )

    if vegtyp is not None:
        state["vegtyp"] = parse_vegetation_type(vegtyp)
        provenance["vegtyp"] = "user override"
    else:
        tvl, tvh = _invariant("tvl"), _invariant("tvh")
        cvl, cvh = _invariant("cvl"), _invariant("cvh")
        if None not in (tvl, tvh, cvl, cvh):
            state["vegtyp"] = map_vegetation_type(tvl, tvh, cvl, cvh)
            provenance["vegtyp"] = (
                f"ERA5 tvl={int(round(tvl))}/tvh={int(round(tvh))} "
                "mapped to IGBP"
            )

    # --- soil profiles -------------------------------------------------
    if has_soil:
        stc = remap_soil_layers(soil_temperature)
        state["stc"] = stc
        provenance["stc"] = "ERA5 stl1-4, remapped to the Noah layers"

        smc_era5 = remap_soil_layers(soil_moisture)
        target_soiltyp = state.get("soiltyp")
        if target_soiltyp is None or era5_soiltyp is None \
                or int(round(era5_soiltyp)) < 1:
            # Without both soil classes the two parameter sets cannot be
            # reconciled, so the volumetric value is carried across unchanged.
            smc = smc_era5
            transfer = "direct volumetric copy, soil classes unavailable"
        else:
            smc = rescale_soil_moisture(
                smc_era5, int(round(era5_soiltyp)), target_soiltyp,
                mode=soil_moisture_transfer,
            )
            transfer = f"{soil_moisture_transfer} transfer to STATSGO {target_soiltyp}"
        state["smc"] = smc
        provenance["smc"] = f"ERA5 swvl1-4, remapped and rescaled ({transfer})"

        if target_soiltyp is not None:
            state["slc"] = liquid_soil_moisture(smc, stc, target_soiltyp)
            provenance["slc"] = "unfrozen fraction of smc, Noah freezing curve"
        else:
            state["slc"] = np.where(stc >= _T_FREEZE, smc, 0.0)
            provenance["slc"] = "smc where unfrozen, no soil class available"

        # The bottom boundary temperature is best represented by the deepest
        # ERA5 layer, which sits below the annual thermal wave at most sites.
        state["tg3"] = float(soil_temperature[-1])
        provenance["tg3"] = "ERA5 stl4"

    # --- skin temperature ----------------------------------------------
    skin = _at_time("skt")
    if skin is not None:
        for name in ("tsfco", "tsfcl", "tisfc"):
            state[name] = skin
            provenance[name] = "ERA5 skt"

    # --- snow -----------------------------------------------------------
    snow_water = _at_time("sd")          # m of water equivalent
    snow_density = _at_time("rsn")       # kg m-3
    if snow_water is not None:
        weasd = snow_water * 1000.0      # mm of water equivalent
        state["weasd"] = weasd
        provenance["weasd"] = "ERA5 sd"
        if snow_density is not None and snow_density > 0.0:
            # Physical depth follows from the water equivalent and the density
            # ratio; both DEPHY fields are in mm.
            state["snowd"] = weasd * 1000.0 / snow_density
            provenance["snowd"] = "ERA5 sd and rsn"
        if "vegtyp" in state:
            state["sncovr"] = snow_cover_fraction(weasd, state["vegtyp"])
            provenance["sncovr"] = "diagnosed from weasd, Noah snfrac"

    # --- canopy water ----------------------------------------------------
    skin_reservoir = _at_time("src")
    if skin_reservoir is not None:
        # ERA5 holds intercepted water as a depth in metres; the DEPHY field is
        # a mass per unit area, and the two are numerically the same to within
        # the density of water.
        state["canopy"] = skin_reservoir * 1000.0
        provenance["canopy"] = "ERA5 src"

    # --- albedo ----------------------------------------------------------
    albedo_pairs = {
        "alvsf": "aluvp", "alvwf": "aluvd",
        "alnsf": "alnip", "alnwf": "alnid",
    }
    for dephy_name, era5_name in albedo_pairs.items():
        value = _at_time(era5_name)
        if value is not None:
            state[dephy_name] = value
            provenance[dephy_name] = f"ERA5 {era5_name}"

    snow_albedo = _at_time("asn")
    if snow_albedo is not None and state.get("weasd", 0.0) > 0.0:
        # ERA5's asn is the instantaneous snow albedo, which only approaches
        # the maximum that snoalb is meant to hold when the snow is fresh, so
        # it is only adopted where there is snow on the ground to describe.
        state["snoalb"] = snow_albedo
        provenance["snoalb"] = "ERA5 asn (instantaneous, snow present)"

    # --- vegetation cover -------------------------------------------------
    cvl, cvh = _invariant("cvl"), _invariant("cvh")
    if cvl is not None and cvh is not None:
        cover = float(np.clip(cvl + cvh, 0.0, 1.0))

        lai_low, lai_high = _at_time("lailv"), _at_time("laihv")
        if lai_low is not None and lai_high is not None:
            # ERA5's cover fractions say how much of the cell is vegetated at
            # all, and are fixed all year.  Noah's vegfrac is the green
            # fraction, which is seasonal, so the cover is scaled by the canopy
            # closure implied by leaf area.  Without this a dormant winter
            # desert and a closed summer canopy both come out near one.
            state["vegfrac"] = float(np.clip(
                cvl * (1.0 - np.exp(-_LAI_EXTINCTION * lai_low))
                + cvh * (1.0 - np.exp(-_LAI_EXTINCTION * lai_high)),
                0.0, 1.0,
            ))
            provenance["vegfrac"] = "ERA5 cvl/cvh closed by lailv/laihv"
        else:
            state["vegfrac"] = cover
            provenance["vegfrac"] = "ERA5 cvl + cvh (no leaf area available)"

        # The cover fraction is the greenness ceiling: it is what vegfrac would
        # reach at full canopy closure.  A true annual range would need a
        # climatology, which is beyond what a single case extraction holds.
        state["shdmax"] = cover
        state["shdmin"] = float(min(0.01, cover))
        provenance["shdmax"] = "ERA5 cvl + cvh, the fully closed canopy limit"
        provenance["shdmin"] = "floor; ERA5 carries no annual greenness cycle"

    # --- roughness ---------------------------------------------------------
    roughness = _at_time("fsr")
    if roughness is not None and roughness > 0.0:
        state["zorl"] = roughness * 100.0  # m to cm
        state["zorll"] = roughness * 100.0
        provenance["zorl"] = "ERA5 fsr"
        provenance["zorll"] = "ERA5 fsr"

    # --- land mask ---------------------------------------------------------
    land_fraction = _invariant("lsm")
    if land_fraction is not None:
        state["slmsk"] = 1.0 if land_fraction >= 0.5 else 0.0
        provenance["slmsk"] = f"ERA5 lsm={land_fraction:.2f}"

    if not state:
        return None

    state["provenance"] = provenance
    return state


def _center_value(field: xr.DataArray, center: int):
    """Select the middle of a stencil, tolerating fields already reduced."""
    selection = {dim: center for dim in ("latitude", "longitude")
                 if dim in field.dims}
    if selection:
        field = field.isel(selection)
    return field.values


def describe_land_state(state: Optional[dict]) -> str:
    """Render a land state as a short human-readable summary."""
    if not state:
        return "  land state: none derived; template values retained"

    lines = []
    vegtyp = state.get("vegtyp")
    soiltyp = state.get("soiltyp")
    if vegtyp is not None:
        lines.append(f"  vegetation: {vegtyp} ({IGBP_CLASS_NAMES[vegtyp]})")
    if soiltyp is not None:
        lines.append(f"  soil:       {soiltyp} ({STATSGO_CLASS_NAMES[soiltyp]})")
    if "stc" in state:
        stc = np.asarray(state["stc"])
        smc = np.asarray(state["smc"])
        lines.append(
            "  soil T:     " + ", ".join(f"{v:.1f}" for v in stc) + " K"
        )
        lines.append(
            "  soil M:     " + ", ".join(f"{v:.3f}" for v in smc) + " m3 m-3"
        )
    if "weasd" in state and state["weasd"] > 0:
        lines.append(f"  snow:       {state['weasd']:.1f} mm w.e.")
    return "\n".join(lines)
