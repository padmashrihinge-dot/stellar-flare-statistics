# ================================================================
# STELLAR FLARE STATISTICS PIPELINE
# TESS + WAVELET + ASTROPY + NUMPY
#
# Project:
# Characterizing Stellar Flare Statistics and Energy Distributions
#
# Pipeline:
# TESS light curve
#       ↓
# Cleaning / normalization
#       ↓
# Savitzky-Golay detrending
#       ↓
# Mexican-hat wavelet transform
#       ↓
# Candidate detection
#       ↓
# Noise + morphology validation
#       ↓
# Flare boundaries
#       ↓
# Equivalent Duration (ED)
#       ↓
# Approximate bolometric-equivalent energy
#       ↓
# Cumulative frequency-energy distribution
#       ↓
# Power-law fitting
#
# No AltaiPony dependency.
# ================================================================

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt

import lightkurve as lk

from astropy.constants import R_sun, sigma_sb
from scipy.signal import savgol_filter, find_peaks


# ================================================================
# SETTINGS
# ================================================================

PROJECT_DIR = r"C:\stellar_flares"
IMAGE_DIR = os.path.join(PROJECT_DIR, "images")

os.makedirs(PROJECT_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

# ------------------------------------------------
# Target stars
# ------------------------------------------------

TARGETS = [
    "EV Lac",
    "AD Leo",
    "YZ CMi",
    "UV Ceti",
    "Wolf 359",
]

# ------------------------------------------------
# IMPORTANT:
# Keep this at 1 for the workshop run.
# It prevents huge downloads and keeps the analysis
# manageable.
# ------------------------------------------------

MAX_OBSERVATIONS_PER_STAR = 1


# ================================================================
# DETECTION PARAMETERS
# ================================================================

# Minimum accepted flare amplitude.
#
# 0.008 = 0.8% relative flux increase.
#
# This is intentionally higher than the previous 0.005
# because the diagnostic plots showed many tiny noise spikes.
#
MIN_FLARE_AMPLITUDE = 0.008


# Wavelet threshold in units of MAD.
#
# Candidate detection:
#
# score > median(score) + WAVELET_MAD_THRESHOLD * MAD
#
WAVELET_MAD_THRESHOLD = 5.0


# Minimum separation between wavelet candidates.
#
# This is applied in units of samples.
# The actual value is calculated from cadence.
#
MIN_CANDIDATE_SEPARATION_MIN = 1.0


# ------------------------------------------------
# Local noise validation
# ------------------------------------------------

# Required peak significance above local noise.
#
# peak excess must be at least:
#
# max(MIN_FLARE_AMPLITUDE,
#     LOCAL_NOISE_SIGMA * local_noise)
#
LOCAL_NOISE_SIGMA = 4.0


# Window used for estimating local noise.
LOCAL_NOISE_WINDOW_MIN = 30.0


# ------------------------------------------------
# Flare duration limits
# ------------------------------------------------

MIN_FLARE_DURATION_MIN = 2.0
MAX_FLARE_DURATION_MIN = 180.0


# ------------------------------------------------
# Flare morphology
# ------------------------------------------------

# A real flare should have some elevated points around
# the peak instead of being just a single isolated spike.
#
MIN_MORPHOLOGY_POINTS = 2

# Fraction of peak amplitude that neighboring points
# should exceed.
MORPHOLOGY_LEVEL = 0.20


# ------------------------------------------------
# Boundary threshold
# ------------------------------------------------

# Walk away from the flare peak until the signal returns
# close to the local baseline.
#
BOUNDARY_FRACTION = 0.10


# ------------------------------------------------
# Detrending
# ------------------------------------------------

# Approximate detrending timescale.
#
# Broad stellar variability is removed while short-duration
# flares remain.
#
DETREND_WINDOW_HOURS = 6.0


# ================================================================
# APPROXIMATE STELLAR PARAMETERS
# ================================================================
#
# These are approximate representative stellar parameters.
#
# The energy calculation therefore represents an
# APPROXIMATE BOLOMETRIC-EQUIVALENT FLARE ENERGY,
# not an exact TESS-band energy.
#
# radius = stellar radius in solar radii
# teff   = effective temperature in Kelvin
# ================================================================

STAR_PARAMETERS = {

    "EV Lac": {
        "radius_rsun": 0.35,
        "teff": 3300,
    },

    "AD Leo": {
        "radius_rsun": 0.43,
        "teff": 3400,
    },

    "YZ CMi": {
        "radius_rsun": 0.30,
        "teff": 3100,
    },

    "UV Ceti": {
        "radius_rsun": 0.14,
        "teff": 3100,
    },

    "Wolf 359": {
        "radius_rsun": 0.16,
        "teff": 2800,
    },
}


# ================================================================
# UTILITY FUNCTIONS
# ================================================================

def robust_mad(values):
    """
    Robust median absolute deviation.
    """

    values = np.asarray(values, dtype=float)

    values = values[np.isfinite(values)]

    if len(values) == 0:
        return 0.0

    med = np.median(values)

    return np.median(np.abs(values - med))


# ------------------------------------------------

def make_odd(value):
    """
    Return an odd integer >= 3.
    """

    value = int(value)

    if value < 3:
        value = 3

    if value % 2 == 0:
        value += 1

    return value


# ------------------------------------------------

def calculate_bolometric_luminosity(star):
    """
    Calculate approximate stellar bolometric luminosity.

    L = 4*pi*R^2*sigma*T^4
    """

    params = STAR_PARAMETERS[star]

    radius = params["radius_rsun"] * R_sun
    temperature = params["teff"]

    luminosity = (
        4
        * np.pi
        * radius.value ** 2
        * sigma_sb.value
        * temperature ** 4
    )

    return luminosity


# ------------------------------------------------

def calculate_flare_energy(ed_seconds, star):
    """
    Approximate bolometric-equivalent flare energy.

    E = ED * L_bol

    Astropy sigma_sb is in SI units, so luminosity is
    initially calculated in watts (J/s).

    Convert the resulting energy from joules to erg:
        1 J = 1e7 erg
    """

    luminosity_watts = calculate_bolometric_luminosity(star)

    energy_joules = ed_seconds * luminosity_watts

    energy_erg = energy_joules * 1e7

    return energy_erg


# ================================================================
# LIGHT CURVE DOWNLOAD
# ================================================================

def download_light_curve(star):
    """
    Search MAST through Lightkurve and download the best
    available short-cadence TESS observation.

    Preference:
    1. Shortest cadence
    2. One observation only
    """

    print(f"\n{star}:")

    try:

        search = lk.search_lightcurve(
            star,
            mission="TESS"
        )

    except Exception as exc:

        print(f"Search failed: {exc}")

        return None


    if len(search) == 0:

        print("No TESS observations found.")

        return None


    print(f"TESS observations found: {len(search)}")


    # ------------------------------------------------
    # Sort by exposure/cadence.
    #
    # Lightkurve's table structure can vary slightly,
    # so we use the search table where possible.
    # ------------------------------------------------

    try:

        exptime = np.asarray(
            search.table["exptime"],
            dtype=float
        )

        order = np.argsort(exptime)

    except Exception:

        order = np.arange(len(search))


    # ------------------------------------------------
    # Try observations in preferred order.
    # ------------------------------------------------

    lc = None

    for idx in order[:MAX_OBSERVATIONS_PER_STAR]:

        try:

            lc = search[idx].download()

            if lc is not None:
                break

        except Exception as exc:

            print(
                f"Could not download observation "
                f"{idx}: {exc}"
            )


    if lc is None:

        print("Could not download a TESS light curve.")

        return None


    return lc


# ================================================================
# CLEAN LIGHT CURVE
# ================================================================

def clean_light_curve(lc):
    """
    Convert the downloaded TESS light curve into clean
    time and normalized flux arrays.
    """

    time = np.asarray(
        lc.time.value,
        dtype=float
    )

    flux = np.asarray(
        lc.flux.value,
        dtype=float
    )


    # Remove non-finite values

    mask = (
        np.isfinite(time)
        & np.isfinite(flux)
        & (flux > 0)
    )

    time = time[mask]
    flux = flux[mask]


    if len(time) < 100:

        raise ValueError(
            "Not enough valid data points."
        )


    # ------------------------------------------------
    # Remove extreme global outliers.
    # ------------------------------------------------

    median_flux = np.median(flux)

    mad_flux = robust_mad(flux)

    if mad_flux > 0:

        robust_sigma = 1.4826 * mad_flux

        mask = np.abs(
            flux - median_flux
        ) < 10 * robust_sigma

        time = time[mask]
        flux = flux[mask]


    # ------------------------------------------------
    # Normalize.
    # ------------------------------------------------

    flux = flux / np.median(flux)


    return time, flux


# ================================================================
# DETRENDING
# ================================================================

def detrend_light_curve(time, flux):
    """
    Remove broad stellar variability with a
    Savitzky-Golay filter.

    Returns:
        detrended flux
        cadence in minutes
    """

    dt_days = np.median(
        np.diff(time)
    )

    cadence_min = dt_days * 24 * 60


    # ------------------------------------------------
    # Window corresponding to approximately 6 hours.
    # ------------------------------------------------

    points_per_window = int(
        DETREND_WINDOW_HOURS * 60
        / cadence_min
    )

    window = make_odd(points_per_window)


    # Make sure the window is not larger than the dataset.

    max_window = len(flux) - 1

    max_window = make_odd(max_window)

    if window > max_window:
        window = max_window


    if window < 5:

        window = 5


    try:

        baseline = savgol_filter(
            flux,
            window_length=window,
            polyorder=2
        )

    except Exception:

        # Fallback for unusual short datasets.

        baseline = np.full_like(
            flux,
            np.median(flux)
        )


    baseline[baseline <= 0] = 1.0


    detrended = flux / baseline


    # Normalize once more around unity.

    detrended = detrended / np.median(detrended)


    return detrended, cadence_min


# ================================================================
# WAVELET CANDIDATE DETECTION
# ================================================================

def wavelet_candidates(
    detrended_flux,
    cadence_min
):
    """
    Detect possible flare peaks using the Mexican-hat
    continuous wavelet transform.

    Returns:
        candidate indices
        wavelet score
    """

    flux = np.asarray(
        detrended_flux,
        dtype=float
    )


    # Remove baseline level.

    signal = flux - np.median(flux)


    # ------------------------------------------------
    # Mexican-hat wavelet.
    #
    # Scales cover short-duration structures.
    # ------------------------------------------------

    scales = np.arange(
        1,
        64
    )


    try:

        coefficients, frequencies = pywt.cwt(
            signal,
            scales,
            "mexh"
        )

    except Exception as exc:

        print(
            f"Wavelet transform failed: {exc}"
        )

        return np.array([], dtype=int), np.zeros_like(flux)


    # ------------------------------------------------
    # Positive wavelet response.
    #
    # We care about positive flare-like excursions.
    # ------------------------------------------------

    positive_coefficients = np.maximum(
        coefficients,
        0
    )


    wavelet_score = np.max(
        positive_coefficients,
        axis=0
    )


    # ------------------------------------------------
    # Robust threshold.
    # ------------------------------------------------

    median_score = np.median(
        wavelet_score
    )

    mad_score = robust_mad(
        wavelet_score
    )


    if mad_score <= 0:

        return np.array([], dtype=int), wavelet_score


    threshold = (
        median_score
        + WAVELET_MAD_THRESHOLD * mad_score
    )


    # ------------------------------------------------
    # Minimum separation.
    # ------------------------------------------------

    min_distance = max(
        1,
        int(
            MIN_CANDIDATE_SEPARATION_MIN
            / cadence_min
        )
    )


    candidates, properties = find_peaks(
        wavelet_score,
        height=threshold,
        distance=min_distance
    )


    return candidates, wavelet_score


# ================================================================
# LOCAL NOISE ESTIMATION
# ================================================================

def local_noise(
    flux,
    index,
    cadence_min
):
    """
    Estimate local noise around a candidate using MAD.
    """

    half_window = max(
        5,
        int(
            LOCAL_NOISE_WINDOW_MIN
            / cadence_min
        )
    )


    start = max(
        0,
        index - half_window
    )

    end = min(
        len(flux),
        index + half_window + 1
    )


    local = flux[start:end]


    if len(local) < 5:
        return 0.0


    # Remove the highest few points so a flare does not
    # dominate the noise estimate.

    local_median = np.median(local)

    deviations = np.abs(
        local - local_median
    )

    cutoff = np.percentile(
        deviations,
        80
    )

    quiet = local[
        deviations <= cutoff
    ]


    noise = 1.4826 * robust_mad(
        quiet
    )


    return noise


# ================================================================
# FLARE BOUNDARIES
# ================================================================

def find_flare_boundaries(
    flux,
    peak_index,
    cadence_min
):
    """
    Determine approximate start and end of a flare.

    Boundaries are based on a fraction of the peak
    excess above the local baseline.
    """

    n = len(flux)


    # ------------------------------------------------
    # Estimate local baseline.
    # ------------------------------------------------

    baseline_window = max(
        10,
        int(
            30.0 / cadence_min
        )
    )


    start_baseline = max(
        0,
        peak_index - baseline_window
    )

    end_baseline = min(
        n,
        peak_index + baseline_window + 1
    )


    local_region = flux[
        start_baseline:end_baseline
    ]


    local_baseline = np.median(
        local_region
    )


    peak_excess = (
        flux[peak_index]
        - local_baseline
    )


    if peak_excess <= 0:

        return None


    boundary_level = (
        local_baseline
        + BOUNDARY_FRACTION * peak_excess
    )


    # ------------------------------------------------
    # Walk left.
    # ------------------------------------------------

    left = peak_index

    while left > 0:

        if flux[left] <= boundary_level:
            break

        left -= 1


    # ------------------------------------------------
    # Walk right.
    # ------------------------------------------------

    right = peak_index

    while right < n - 1:

        if flux[right] <= boundary_level:
            break

        right += 1


    duration_min = (
        (right - left)
        * cadence_min
    )


    if duration_min < MIN_FLARE_DURATION_MIN:
        return None


    if duration_min > MAX_FLARE_DURATION_MIN:
        return None


    return left, right, local_baseline


# ================================================================
# MORPHOLOGY VALIDATION
# ================================================================

def morphology_valid(
    flux,
    peak_index,
    left,
    right,
    baseline
):
    """
    Reject isolated one-point spikes.

    A candidate is accepted only if there are elevated
    points around the peak.
    """

    peak_excess = (
        flux[peak_index]
        - baseline
    )


    if peak_excess <= 0:
        return False


    level = (
        baseline
        + MORPHOLOGY_LEVEL * peak_excess
    )


    # ------------------------------------------------
    # Count elevated points before the peak.
    # ------------------------------------------------

    left_region = flux[left:peak_index]

    right_region = flux[
        peak_index + 1:right + 1
    ]


    left_count = np.sum(
        left_region > level
    )

    right_count = np.sum(
        right_region > level
    )


    # ------------------------------------------------
    # We require some support on both sides.
    #
    # This rejects many isolated noise spikes.
    # ------------------------------------------------

    if (
        left_count < MIN_MORPHOLOGY_POINTS
        and right_count < MIN_MORPHOLOGY_POINTS
    ):

        return False


    return True


# ================================================================
# FLARE VALIDATION
# ================================================================

def validate_candidates(
    time,
    detrended_flux,
    candidates,
    cadence_min
):
    """
    Apply amplitude, local-noise and morphology validation.
    """

    accepted = []


    # ------------------------------------------------
    # Slight smoothing ONLY for validation.
    #
    # Original detrended flux is still used for ED.
    # ------------------------------------------------

    smooth_window = make_odd(5)

    if smooth_window >= len(detrended_flux):
        smooth_window = make_odd(
            max(
                3,
                len(detrended_flux) - 1
            )
        )


    try:

        validation_flux = savgol_filter(
            detrended_flux,
            window_length=smooth_window,
            polyorder=2
        )

    except Exception:

        validation_flux = detrended_flux.copy()


    for peak_index in candidates:

        # --------------------------------------------
        # Use the smoothed signal to validate peak.
        # --------------------------------------------

        peak_flux = validation_flux[
            peak_index
        ]


        # --------------------------------------------
        # Local baseline.
        # --------------------------------------------

        window = max(
            10,
            int(
                LOCAL_NOISE_WINDOW_MIN
                / cadence_min
            )
        )


        s = max(
            0,
            peak_index - window
        )

        e = min(
            len(validation_flux),
            peak_index + window + 1
        )


        local_region = validation_flux[s:e]

        baseline = np.median(
            local_region
        )


        peak_excess = (
            peak_flux - baseline
        )


        # --------------------------------------------
        # Amplitude requirement.
        # --------------------------------------------

        if peak_excess < MIN_FLARE_AMPLITUDE:
            continue


        # --------------------------------------------
        # Local noise requirement.
        # --------------------------------------------

        noise = local_noise(
            validation_flux,
            peak_index,
            cadence_min
        )


        if noise > 0:

            if peak_excess < (
                LOCAL_NOISE_SIGMA * noise
            ):

                continue


        # --------------------------------------------
        # Find flare boundaries on ORIGINAL
        # detrended flux.
        # --------------------------------------------

        boundaries = find_flare_boundaries(
            detrended_flux,
            peak_index,
            cadence_min
        )


        if boundaries is None:
            continue


        left, right, true_baseline = boundaries


        # --------------------------------------------
        # Morphology validation.
        # --------------------------------------------

        if not morphology_valid(
            validation_flux,
            peak_index,
            left,
            right,
            baseline
        ):

            continue


        # --------------------------------------------
        # Store accepted flare.
        # --------------------------------------------

        accepted.append({
            "peak_index": int(peak_index),
            "start_index": int(left),
            "end_index": int(right),
            "baseline": float(true_baseline),
            "peak_flux": float(
                detrended_flux[peak_index]
            ),
            "amplitude": float(
                detrended_flux[peak_index]
                - true_baseline
            ),
            "local_noise": float(noise),
        })


    return accepted


# ================================================================
# MERGE OVERLAPPING FLARES
# ================================================================

def merge_overlapping_flares(
    flares,
    cadence_min
):
    """
    Merge detections that belong to the same flare.

    This is useful when multiple wavelet scales detect
    the same physical event.
    """

    if not flares:
        return []


    flares = sorted(
        flares,
        key=lambda x: x["peak_index"]
    )


    merged = [flares[0].copy()]


    merge_gap_points = max(
        1,
        int(
            2.0 / cadence_min
        )
    )


    for current in flares[1:]:

        previous = merged[-1]


        # ------------------------------------------------
        # Overlap or very close events.
        # ------------------------------------------------

        if (
            current["start_index"]
            <= previous["end_index"]
            + merge_gap_points
        ):

            # Keep the wider boundary.

            previous["start_index"] = min(
                previous["start_index"],
                current["start_index"]
            )

            previous["end_index"] = max(
                previous["end_index"],
                current["end_index"]
            )


            # Keep the stronger peak.

            if (
                current["amplitude"]
                > previous["amplitude"]
            ):

                previous["peak_index"] = (
                    current["peak_index"]
                )

                previous["peak_flux"] = (
                    current["peak_flux"]
                )

                previous["amplitude"] = (
                    current["amplitude"]
                )

                previous["local_noise"] = (
                    current["local_noise"]
                )

        else:

            merged.append(
                current.copy()
            )


    return merged


# ================================================================
# EQUIVALENT DURATION
# ================================================================

def calculate_equivalent_duration(
    time,
    flux,
    flare,
    cadence_min
):
    """
    Calculate equivalent duration.

    ED = integral[(F - F_baseline) / F_baseline] dt

    Result is in seconds.
    """

    left = flare["start_index"]
    right = flare["end_index"]


    segment_time = time[
        left:right + 1
    ]


    segment_flux = flux[
        left:right + 1
    ]


    baseline = flare["baseline"]


    if baseline <= 0:
        return 0.0


    relative_excess = (
        segment_flux - baseline
    ) / baseline


    # Do not allow negative contribution
    # inside the flare interval.

    relative_excess = np.maximum(
        relative_excess,
        0
    )


    # Convert TESS days to seconds.

    segment_seconds = (
        segment_time
        - segment_time[0]
    ) * 86400.0


    ed_seconds = np.trapezoid(
        relative_excess,
        segment_seconds
    )


    return max(
        0.0,
        float(ed_seconds)
    )


# ================================================================
# PROCESS ONE STAR
# ================================================================

def process_star(star):
    """
    Complete analysis for one target.
    """

    lc = download_light_curve(star)


    if lc is None:
        return [], None


    try:

        time, flux = clean_light_curve(lc)

    except Exception as exc:

        print(
            f"Cleaning failed for {star}: {exc}"
        )

        return [], None


    try:

        detrended, cadence_min = (
            detrend_light_curve(
                time,
                flux
            )
        )

    except Exception as exc:

        print(
            f"Detrending failed for {star}: {exc}"
        )

        return [], None


    print(
        f"Cadence: {cadence_min:.2f} minutes"
    )


    # ============================================================
    # WAVELET DETECTION
    # ============================================================

    candidates, wavelet_score = (
        wavelet_candidates(
            detrended,
            cadence_min
        )
    )


    print(
        f"Wavelet candidates: "
        f"{len(candidates)}"
    )


    # ============================================================
    # VALIDATION
    # ============================================================

    validated = validate_candidates(
        time,
        detrended,
        candidates,
        cadence_min
    )


    # ============================================================
    # MERGE DUPLICATE / OVERLAPPING EVENTS
    # ============================================================

    validated = merge_overlapping_flares(
        validated,
        cadence_min
    )


    print(
        f"Validated flares: "
        f"{len(validated)}"
    )


    # ============================================================
    # CALCULATE ED + ENERGY
    # ============================================================

    flare_rows = []


    for flare_number, flare in enumerate(
        validated,
        start=1
    ):

        ed_seconds = (
            calculate_equivalent_duration(
                time,
                detrended,
                flare,
                cadence_min
            )
        )


        energy = calculate_flare_energy(
            ed_seconds,
            star
        )


        start_idx = flare["start_index"]
        peak_idx = flare["peak_index"]
        end_idx = flare["end_index"]


        start_time = time[start_idx]
        peak_time = time[peak_idx]
        end_time = time[end_idx]


        duration_min = (
            end_time - start_time
        ) * 24 * 60


        flare_rows.append({

            "star": star,

            "flare_id": (
                f"{star.replace(' ', '_')}"
                f"_{flare_number:03d}"
            ),

            "start_btjd": start_time,

            "peak_btjd": peak_time,

            "end_btjd": end_time,

            "duration_min": duration_min,

            "peak_flux": flare["peak_flux"],

            "amplitude": flare["amplitude"],

            "local_noise": flare["local_noise"],

            "equivalent_duration_s": (
                ed_seconds
            ),

            "approx_energy_erg": (
                energy
            ),
        })


    # ============================================================
    # CREATE DIAGNOSTIC PLOT
    # ============================================================

    create_diagnostic_plot(
        star,
        time,
        flux,
        detrended,
        candidates,
        validated
    )


    return flare_rows, {
        "time": time,
        "flux": flux,
        "detrended": detrended,
        "candidates": candidates,
        "validated": validated,
        "cadence_min": cadence_min,
    }


# ================================================================
# DIAGNOSTIC PLOT
# ================================================================

def create_diagnostic_plot(
    star,
    time,
    flux,
    detrended,
    candidates,
    flares
):
    """
    Create the diagnostic plot showing:
    TESS flux
    detrended flux
    wavelet candidates
    accepted flare regions
    """

    plt.figure(
        figsize=(18, 7)
    )


    # ------------------------------------------------
    # Original TESS flux
    # ------------------------------------------------

    plt.plot(
        time,
        flux,
        linewidth=0.8,
        alpha=0.55,
        label="TESS flux"
    )


    # ------------------------------------------------
    # Detrended flux
    # ------------------------------------------------

    plt.plot(
        time,
        detrended,
        linewidth=0.8,
        label="Detrended flux"
    )


    # ------------------------------------------------
    # Wavelet candidates
    # ------------------------------------------------

    if len(candidates) > 0:

        plt.scatter(
            time[candidates],
            detrended[candidates],
            marker="^",
            s=45,
            label="Wavelet candidates"
        )


    # ------------------------------------------------
    # Accepted flare regions
    # ------------------------------------------------

    first_region = True

    for flare in flares:

        left = flare["start_index"]
        right = flare["end_index"]

        plt.axvspan(
            time[left],
            time[right],
            alpha=0.20,
            label=(
                "Detected flare"
                if first_region
                else None
            )
        )

        first_region = False


    plt.title(
        f"{star} — Automated Wavelet Flare Detection",
        fontsize=16
    )


    plt.xlabel(
        "TESS time [BTJD]"
    )


    plt.ylabel(
        "Normalized flux"
    )


    plt.grid(
        alpha=0.25
    )


    plt.legend()


    plt.tight_layout()


    filename = os.path.join(
        IMAGE_DIR,
        f"{star.replace(' ', '_')}_wavelet_detection.png"
    )


    plt.savefig(filename, dpi=600, bbox_inches="tight", facecolor="white")


    plt.close()


# ================================================================
# CLEAN FINAL FLARE PLOT
# ================================================================

def create_final_flare_plot(
    star,
    time,
    detrended,
    flares
):
    """
    Cleaner scientific plot showing only accepted flares.
    """

    plt.figure(
        figsize=(18, 7)
    )


    plt.plot(
        time,
        detrended,
        linewidth=0.8,
        label="Detrended flux"
    )


    first = True


    for flare in flares:

        left = flare["start_index"]
        right = flare["end_index"]
        peak = flare["peak_index"]


        plt.axvspan(
            time[left],
            time[right],
            alpha=0.20,
            label=(
                "Accepted flare"
                if first
                else None
            )
        )


        plt.scatter(
            time[peak],
            detrended[peak],
            marker="^",
            s=50
        )


        first = False


    plt.axhline(
        1.0,
        linestyle="--",
        linewidth=0.8,
        alpha=0.6
    )


    plt.title(
        f"{star} — Validated Stellar Flares",
        fontsize=16
    )


    plt.xlabel(
        "TESS time [BTJD]"
    )


    plt.ylabel(
        "Normalized detrended flux"
    )


    plt.grid(
        alpha=0.25
    )


    plt.legend()


    plt.tight_layout()


    filename = os.path.join(
        IMAGE_DIR,
        f"{star.replace(' ', '_')}_validated_flares.png"
    )


    plt.savefig(filename, dpi=600, bbox_inches="tight", facecolor="white")


    plt.close()


# ================================================================
# POWER-LAW ANALYSIS
# ================================================================

def power_law_analysis(
    star,
    flare_df
):
    """
    Calculate cumulative frequency-energy distribution
    and power-law slope.

    Differential distribution:

        dN/dE ∝ E^-alpha

    Cumulative distribution:

        N(>E) ∝ E^-(alpha - 1)

    IMPORTANT:
    The energy threshold used here is a preliminary
    lower-energy cut, NOT an injection-recovery
    completeness limit.
    """

    if flare_df.empty:

        return None


    energies = flare_df[
        "approx_energy_erg"
    ].values.astype(float)


    energies = energies[
        np.isfinite(energies)
        & (energies > 0)
    ]


    if len(energies) < 5:

        print(
            "Not enough flares for power-law analysis."
        )

        return None


    # ------------------------------------------------
    # Preliminary lower-energy cut.
    #
    # We deliberately do NOT call this a definitive
    # completeness limit.
    # ------------------------------------------------

    energy_threshold = np.percentile(
        energies,
        25
    )


    used = energies[
        energies >= energy_threshold
    ]


    if len(used) < 5:

        return None


    # ============================================================
    # DIFFERENTIAL POWER-LAW MLE
    # ============================================================

    log_ratio = np.log(
        used / energy_threshold
    )


    denominator = np.sum(
        log_ratio
    )


    if denominator <= 0:

        return None


    alpha = (
        1.0
        + len(used)
        / denominator
    )


    # ============================================================
    # CUMULATIVE DISTRIBUTION
    # ============================================================

    sorted_energy = np.sort(
        used
    )


    n = len(sorted_energy)


    cumulative_number = np.arange(
        n,
        0,
        -1
    )


    log_energy = np.log10(
        sorted_energy
    )

    log_cumulative = np.log10(
        cumulative_number
    )


    # ============================================================
    # CUMULATIVE SLOPE
    # ============================================================

    slope, intercept = np.polyfit(
        log_energy,
        log_cumulative,
        1
    )


    predicted = (
        slope * log_energy
        + intercept
    )


    ss_res = np.sum(
        (log_cumulative - predicted) ** 2
    )


    ss_tot = np.sum(
        (
            log_cumulative
            - np.mean(log_cumulative)
        ) ** 2
    )


    if ss_tot > 0:

        r_squared = (
            1
            - ss_res / ss_tot
        )

    else:

        r_squared = np.nan


    print(
        "Power-law:"
    )

    print(
        f"  Flares: {len(energies)}"
    )

    print(
        f"  Used: {len(used)}"
    )

    print(
        f"  Energy threshold: "
        f"{energy_threshold:.3e} erg"
    )

    print(
        f"  Alpha: {alpha:.4f}"
    )

    print(
        f"  Cumulative slope: "
        f"{slope:.4f}"
    )

    print(
        f"  R²: {r_squared:.4f}"
    )


    # ============================================================
    # POWER-LAW PLOT
    # ============================================================

    plt.figure(
        figsize=(9, 7)
    )


    plt.loglog(
        sorted_energy,
        cumulative_number,
        "o",
        markersize=5,
        label="Observed flares"
    )


    fit_energy = np.logspace(
        np.log10(sorted_energy.min()),
        np.log10(sorted_energy.max()),
        100
    )


    fit_number = (
        10 ** intercept
    ) * (
        fit_energy ** slope
    )


    plt.loglog(
        fit_energy,
        fit_number,
        linewidth=2,
        label=(
            f"Power-law fit "
            f"(α = {alpha:.2f})"
        )
    )


    plt.axvline(
        energy_threshold,
        linestyle="--",
        linewidth=1,
        alpha=0.7,
        label="Preliminary lower-energy cut"
    )


    plt.xlabel(
        "Approximate flare energy [erg]"
    )


    plt.ylabel(
        "Cumulative number N(>E)"
    )


    plt.title(
        f"{star} — Cumulative Flare Energy Distribution"
    )


    plt.grid(
        alpha=0.25,
        which="both"
    )


    plt.legend()


    plt.tight_layout()


    filename = os.path.join(
        IMAGE_DIR,
        f"{star.replace(' ', '_')}_power_law.png"
    )


    plt.savefig(filename, dpi=600, bbox_inches="tight", facecolor="white")


    plt.close()


    return {
        "star": star,

        "total_flares": len(energies),

        "used_flares": len(used),

        "energy_threshold_erg": (
            energy_threshold
        ),

        "alpha": alpha,

        "cumulative_slope": slope,

        "r_squared": r_squared,
    }


# ================================================================
# COMPARISON PLOT
# ================================================================

def create_comparison_plot(
    results_df
):
    """
    Compare cumulative power-law distributions
    for all stars.
    """

    if results_df.empty:
        return


    plt.figure(
        figsize=(10, 7)
    )


    for _, row in results_df.iterrows():

        star = row["star"]

        energies = row.get(
            "_energies",
            None
        )


        if energies is None:
            continue


        energies = np.asarray(
            energies,
            dtype=float
        )


        energies = energies[
            np.isfinite(energies)
            & (energies > 0)
        ]


        if len(energies) < 5:
            continue


        threshold = np.percentile(
            energies,
            25
        )


        used = energies[
            energies >= threshold
        ]


        if len(used) < 5:
            continue


        used = np.sort(
            used
        )


        cumulative = np.arange(
            len(used),
            0,
            -1
        )


        plt.loglog(
            used,
            cumulative,
            "o-",
            markersize=4,
            linewidth=1,
            label=star
        )


    plt.xlabel(
        "Approximate flare energy [erg]"
    )


    plt.ylabel(
        "Cumulative number N(>E)"
    )


    plt.title(
        "Comparison of Stellar Flare Energy Distributions"
    )


    plt.grid(
        alpha=0.25,
        which="both"
    )


    plt.legend()


    plt.tight_layout()


    filename = os.path.join(
        IMAGE_DIR,
        "All_Stars_Power_Law_Comparison.png"
    )


    plt.savefig(filename, dpi=600, bbox_inches="tight", facecolor="white")


    plt.close()


# ================================================================
# MAIN PROGRAM
# ================================================================

def main():

    print()
    print("=" * 70)
    print(
        " STELLAR FLARE STATISTICS PIPELINE"
    )
    print(
        " TESS + WAVELET + ASTROPY + NUMPY"
    )
    print("=" * 70)


    all_flares = []

    star_results = []

    energy_storage = []


    # ============================================================
    # PROCESS ALL STARS
    # ============================================================

    for star in TARGETS:

        try:

            flare_rows, analysis = process_star(
                star
            )

        except Exception as exc:

            print()
            print(
                f"ERROR processing {star}:"
            )
            print(exc)

            continue


        # --------------------------------------------------------
        # Store flare catalog
        # --------------------------------------------------------

        all_flares.extend(
            flare_rows
        )


        # --------------------------------------------------------
        # Final clean plot
        # --------------------------------------------------------

        if analysis is not None:

            create_final_flare_plot(
                star,
                analysis["time"],
                analysis["detrended"],
                analysis["validated"]
            )


        # --------------------------------------------------------
        # Power-law analysis
        # --------------------------------------------------------

        if len(flare_rows) > 0:

            flare_df_star = pd.DataFrame(
                flare_rows
            )


            power_result = (
                power_law_analysis(
                    star,
                    flare_df_star
                )
            )


            if power_result is not None:

                power_result["_energies"] = (
                    flare_df_star[
                        "approx_energy_erg"
                    ].values
                )


                star_results.append(
                    power_result
                )


        print()


    # ============================================================
    # SAVE FLARE CATALOG
    # ============================================================

    flare_catalog = pd.DataFrame(
        all_flares
    )


    catalog_file = os.path.join(
        PROJECT_DIR,
        "UV_Ceti_TESS_Flare_Catalog.csv"
    )


    if not flare_catalog.empty:

        flare_catalog.to_csv(
            catalog_file,
            index=False
        )


    # ============================================================
    # STAR SUMMARY
    # ============================================================

    summary_rows = []


    for star in TARGETS:

        star_flares = flare_catalog[
            flare_catalog["star"] == star
        ] if not flare_catalog.empty else pd.DataFrame()


        count = len(
            star_flares
        )


        if count > 0:

            total_ed = star_flares[
                "equivalent_duration_s"
            ].sum()


            median_energy = np.median(
                star_flares[
                    "approx_energy_erg"
                ]
            )


            mean_energy = np.mean(
                star_flares[
                    "approx_energy_erg"
                ]
            )


            maximum_energy = np.max(
                star_flares[
                    "approx_energy_erg"
                ]
            )


        else:

            total_ed = 0
            median_energy = np.nan
            mean_energy = np.nan
            maximum_energy = np.nan


        params = STAR_PARAMETERS[
            star
        ]


        summary_rows.append({

            "star": star,

            "number_of_validated_flares": count,

            "radius_rsun": (
                params["radius_rsun"]
            ),

            "teff_K": (
                params["teff"]
            ),

            "bolometric_luminosity_erg_s": (
                calculate_bolometric_luminosity(
                    star
                )
            ),

            "total_equivalent_duration_s": (
                total_ed
            ),

            "median_flare_energy_erg": (
                median_energy
            ),

            "mean_flare_energy_erg": (
                mean_energy
            ),

            "maximum_flare_energy_erg": (
                maximum_energy
            ),
        })


    summary_df = pd.DataFrame(
        summary_rows
    )


    summary_file = os.path.join(
        PROJECT_DIR,
        "UV_Ceti_Star_Summary.csv"
    )


    summary_df.to_csv(
        summary_file,
        index=False
    )


    # ============================================================
    # POWER-LAW RESULTS TABLE
    # ============================================================

    if star_results:

        power_output_rows = []


        for result in star_results:

            power_output_rows.append({

                "star": result["star"],

                "total_flares": (
                    result["total_flares"]
                ),

                "used_flares": (
                    result["used_flares"]
                ),

                "preliminary_lower_energy_cut_erg": (
                    result[
                        "energy_threshold_erg"
                    ]
                ),

                "alpha": (
                    result["alpha"]
                ),

                "cumulative_slope": (
                    result["cumulative_slope"]
                ),

                "r_squared": (
                    result["r_squared"]
                ),
            })


        power_df = pd.DataFrame(
            power_output_rows
        )


    else:

        power_df = pd.DataFrame(
            columns=[
                "star",
                "total_flares",
                "used_flares",
                "preliminary_lower_energy_cut_erg",
                "alpha",
                "cumulative_slope",
                "r_squared",
            ]
        )


    power_file = os.path.join(
        PROJECT_DIR,
        "UV_Ceti_Power_Law_Results.csv"
    )


    power_df.to_csv(
        power_file,
        index=False
    )


    # ============================================================
    # COMPARISON PLOT
    # ============================================================

    if star_results:

        comparison_df = pd.DataFrame(
            star_results
        )


        create_comparison_plot(
            comparison_df
        )


    # ============================================================
    # FINAL SUMMARY
    # ============================================================

    print()
    print("=" * 70)
    print(
        " FINAL RESULTS"
    )
    print("=" * 70)


    if not power_df.empty:

        display_df = power_df[
            [
                "star",
                "total_flares",
                "used_flares",
                "alpha",
                "cumulative_slope",
                "r_squared",
            ]
        ].copy()


        print(
            display_df.to_string(
                index=False,
                float_format=lambda x:
                f"{x:.4f}"
            )
        )


    print()
    print(
        f"Total validated flares: "
        f"{len(flare_catalog)}"
    )


    print()
    print(
        "CSV files:"
    )

    print(
        catalog_file
    )

    print(
        summary_file
    )

    print(
        power_file
    )


    print()
    print(
        "PNG files:"
    )

    print(
        IMAGE_DIR
    )


    print()
    print("=" * 70)
    print(
        " ANALYSIS COMPLETE"
    )
    print("=" * 70)


# ================================================================
# RUN
# ================================================================

if __name__ == "__main__":

    # Suppress only harmless Lightkurve warnings.
    warnings.filterwarnings(
        "ignore",
        message=".*tpfmodel.*"
    )

    main()