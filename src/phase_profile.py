from __future__ import annotations

"""Modified Bourgouin precipitation-type engine.

Implements the core probabilistic relationships described by Birk et al. (2021):
wet-bulb melting/refreezing energy, heterogeneous ice-nucleation probability,
and probabilities for RA/SN/FZRA/PL.

Inputs are pressure levels plus collocated wet-bulb temperature, model
geopotential height, temperature, and RH with respect to ice.
"""

import numpy as np

G = 9.80665
T0_K = 273.15


def _clip(a: np.ndarray, lo: float = 0.0, hi: float = 100.0) -> np.ndarray:
    return np.clip(np.asarray(a, dtype=np.float32), lo, hi)


def _layer_energy(tw0: np.ndarray, tw1: np.ndarray, z0: np.ndarray, z1: np.ndarray, positive: bool) -> np.ndarray:
    """Energy contribution of one layer, splitting a zero crossing linearly."""
    tw0 = np.asarray(tw0, dtype=np.float32)
    tw1 = np.asarray(tw1, dtype=np.float32)
    z0 = np.asarray(z0, dtype=np.float32)
    z1 = np.asarray(z1, dtype=np.float32)

    out = np.zeros_like(tw0, dtype=np.float32)
    finite = np.isfinite(tw0) & np.isfinite(tw1) & np.isfinite(z0) & np.isfinite(z1)
    dz = np.maximum(z1 - z0, 0.0)
    finite &= dz > 0
    if not np.any(finite):
        return out

    # Linear profile through the layer. For a crossing, integrate the triangular
    # positive/negative portion only.
    same = (tw0 >= 0) & (tw1 >= 0) if positive else (tw0 <= 0) & (tw1 <= 0)
    m = finite & same
    if np.any(m):
        mean_abs = 0.5 * (np.abs(tw0[m]) + np.abs(tw1[m]))
        out[m] = G * mean_abs * dz[m] / T0_K

    crossing = finite & ~same & ((tw0 > 0) | (tw1 > 0)) & ((tw0 < 0) | (tw1 < 0))
    if np.any(crossing):
        frac = np.abs(tw0[crossing]) / (np.abs(tw0[crossing]) + np.abs(tw1[crossing]) + 1e-6)
        z_cross = z0[crossing] + frac * dz[crossing]
        if positive:
            # Positive side is from z_cross to the warm endpoint.
            warm_dz = np.where(tw1[crossing] > 0, z1[crossing] - z_cross, z_cross - z0[crossing])
            warm_t = np.where(tw1[crossing] > 0, tw1[crossing], tw0[crossing])
        else:
            cold_dz = np.where(tw1[crossing] < 0, z1[crossing] - z_cross, z_cross - z0[crossing])
            warm_dz = cold_dz
            warm_t = np.where(tw1[crossing] < 0, tw1[crossing], tw0[crossing])
        out[crossing] = G * 0.5 * np.abs(warm_t) * np.maximum(warm_dz, 0.0) / T0_K

    return out


def _prob_ice(temp_c: np.ndarray, rh_ice: np.ndarray, height_m: np.ndarray) -> np.ndarray:
    """Approximate Birk et al. precipitation-generation-layer ProbIce."""
    lev, ny, nx = temp_c.shape
    result = np.zeros((ny, nx), dtype=np.float32)

    # Search contiguous saturated layers from each level upward. A layer must
    # be at least 1 km deep and have RH_ice >= 75%. The minimum temperature in
    # the qualifying layer sets ProbIce.
    for k0 in range(lev - 1):
        z0 = height_m[k0]
        for k1 in range(k0 + 1, lev):
            z1 = height_m[k1]
            depth = z1 - z0
            if depth is None:
                continue
            mask = (
                np.isfinite(depth)
                & (depth >= 1000.0)
                & np.all(np.isfinite(temp_c[k0:k1 + 1]), axis=0)
                & np.all(np.isfinite(rh_ice[k0:k1 + 1]), axis=0)
                & np.all(rh_ice[k0:k1 + 1] >= 75.0, axis=0)
            )
            if not np.any(mask):
                continue
            min_t = np.nanmin(np.where(mask[None, ...], temp_c[k0:k1 + 1], np.nan), axis=0)
            p = np.zeros_like(result)
            cold15 = min_t <= -15.0
            mid = (min_t > -15.0) & (min_t < -7.0)
            p[cold15] = 100.0
            t = min_t[mid]
            p[mid] = -0.065 * t**4 - 3.1544 * t**3 - 56.414 * t**2 - 449.6 * t - 1308.0
            result = np.maximum(result, _clip(p))
    return result


def classify_from_vertical_profile(
    pressure_hpa: np.ndarray,
    wetbulb_c: np.ndarray,
    height_m: np.ndarray,
    temperature_c: np.ndarray,
    rh_ice_pct: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return Modified Bourgouin RA/SN/FZRA/PL probabilities.

    Arrays are [level, y, x]. Pressure levels must correspond to the first
    dimension. Heights must increase with level (low pressure upward).
    """
    p = np.asarray(pressure_hpa, dtype=np.float32)
    tw = np.asarray(wetbulb_c, dtype=np.float32)
    z = np.asarray(height_m, dtype=np.float32)
    temp = np.asarray(temperature_c, dtype=np.float32)
    rhice = np.asarray(rh_ice_pct, dtype=np.float32)
    if tw.ndim != 3 or z.shape != tw.shape or temp.shape != tw.shape or rhice.shape != tw.shape:
        raise ValueError("Profile arrays must all have shape [level, y, x].")
    if p.ndim != 1 or p.size != tw.shape[0]:
        raise ValueError("pressure_hpa must be a 1-D array matching profile levels.")

    # Sort low pressure/high altitude last? We want height increasing.
    order = np.argsort(p)[::-1]
    p = p[order]
    tw = tw[order]
    z = z[order]
    temp = temp[order]
    rhice = rhice[order]

    shape = tw.shape[1:]
    me_layers = []
    re_layers = []
    for k in range(len(p) - 1):
        me_layers.append(_layer_energy(tw[k], tw[k + 1], z[k], z[k + 1], True))
        re_layers.append(_layer_energy(tw[k], tw[k + 1], z[k], z[k + 1], False))
    me_layers = np.stack(me_layers)
    re_layers = np.stack(re_layers)
    me_total = np.nansum(me_layers, axis=0)

    # Find the first elevated warm layer above the surface. RE relevant to
    # PL/FZRA is the negative energy beneath the lowest meaningful warm layer.
    surface_tw = tw[0]
    warm_layer = np.any(tw > 0.0, axis=0)
    lowest_warm_k = np.zeros(shape, dtype=np.int16)
    found = np.zeros(shape, dtype=bool)
    for k in range(len(p)):
        m = (~found) & (tw[k] > 0.0)
        lowest_warm_k[m] = k
        found[m] = True
    re_surface = np.zeros(shape, dtype=np.float32)
    for k in range(len(p) - 1):
        # A negative layer contributes when it is beneath the lowest warm level.
        m = warm_layer & (k < lowest_warm_k)
        re_surface[m] += re_layers[k][m]

    prob_ice = _prob_ice(temp, rhice, z)

    prob_sn_i = _clip(1540.0 * np.exp(-0.29 * me_total))
    prob_sn = _clip((prob_ice / 100.0) * prob_sn_i)

    # Default rain probability. For a purely warm/surface-based melting
    # environment, the paper's FZRA probability becomes RA when surface Tw > 0.
    prob_fzra_i = _clip(-2.1 * re_surface + 0.2 * me_total + 458.0)
    low_me = me_total < 5.0
    prob_fzra_i[low_me] *= 0.2 * me_total[low_me]
    prob_fzra = _clip((100.0 - prob_ice) + (prob_ice / 100.0) * prob_fzra_i)

    prob_pl_i = _clip(2.3 * re_surface - 42.0 * np.log1p(me_total) + 3.0)
    prob_pl = _clip((prob_ice / 100.0) * prob_pl_i)

    # If there is no meaningful melting energy, treat the column as snow/ice
    # rather than forcing the elevated-melt equations to manufacture FZRA/PL.
    no_melt = me_total < 2.0
    prob_sn[no_melt] = np.maximum(prob_sn[no_melt], 100.0)
    prob_pl[no_melt] = 0.0
    prob_fzra[no_melt] = 0.0

    # Surface-based warm layer: rain replaces freezing rain. Keep snow as a
    # competing probability where melting energy is weak.
    surface_warm = surface_tw > 0.0
    prob_rain = np.zeros(shape, dtype=np.float32)
    prob_rain[surface_warm] = prob_fzra[surface_warm]
    prob_fzra[surface_warm] = 0.0

    # Cold surface + elevated melt => no direct rain unless the refreezing
    # layer is absent and the surface is near/above freezing.
    near_surface_warm = surface_tw > -0.25
    prob_rain[near_surface_warm & ~surface_warm] = np.maximum(
        prob_rain[near_surface_warm & ~surface_warm],
        0.25 * prob_fzra[near_surface_warm & ~surface_warm],
    )

    # A column with a surface-based warm layer and no meaningful ice chance is
    # overwhelmingly rain; otherwise the probabilistic values are retained.
    prob_rain = _clip(prob_rain)

    return {
        "snow": prob_sn.astype(np.float32),
        "sleet": prob_pl.astype(np.float32),
        "freezing_rain": prob_fzra.astype(np.float32),
        "rain": prob_rain.astype(np.float32),
        "melting_energy": me_total.astype(np.float32),
        "refreezing_energy": re_surface.astype(np.float32),
        "prob_ice": prob_ice.astype(np.float32),
    }


def probabilities_to_phase(probabilities: dict[str, np.ndarray], winter_threshold: float = 55.0) -> tuple[np.ndarray, np.ndarray]:
    """Conservatively convert PoWT to the existing map phase/confidence fields."""
    from classifier import CLEAR, FZRA, MIXED, RAIN, SLEET, SNOW, UNKNOWN

    names = ["rain", "snow", "sleet", "freezing_rain"]
    stack = np.stack([np.asarray(probabilities[n], dtype=np.float32) for n in names], axis=0)
    order = np.argsort(stack, axis=0)
    top = order[-1]
    second = order[-2]
    topv = np.take_along_axis(stack, top[None, ...], axis=0)[0]
    secondv = np.take_along_axis(stack, second[None, ...], axis=0)[0]

    phase = np.full(top.shape, UNKNOWN, dtype=np.uint8)
    confidence = np.zeros(top.shape, dtype=np.float32)

    # No meaningful probability information.
    valid = np.isfinite(stack).any(axis=0)
    phase[~valid] = CLEAR

    # Rain only becomes the transparent category when it is clearly dominant.
    rain = valid & (top == 0) & (topv >= winter_threshold) & ((topv - secondv) >= 10.0)
    phase[rain] = RAIN
    confidence[rain] = np.minimum(topv[rain] / 100.0, 1.0)

    # Strong winter phase.
    for idx, code in ((1, SNOW), (2, SLEET), (3, FZRA)):
        m = valid & (top == idx) & (topv >= winter_threshold) & ((topv - secondv) >= 10.0)
        phase[m] = code
        confidence[m] = np.minimum(topv[m] / 100.0, 1.0)

    # Close competing precipitation types are deliberately shown as mixed
    # rather than pretending the profile supports a single phase. This includes
    # rain/winter transitions as well as winter/winter mixtures.
    mixed = valid & (topv >= 35.0) & (secondv >= 35.0) & ((topv - secondv) < 15.0)
    phase[mixed] = MIXED
    confidence[mixed] = np.minimum((topv[mixed] + secondv[mixed]) / 200.0, 1.0)

    return phase, confidence
