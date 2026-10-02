# Characterizing Stellar Flare Statistics and Energy Distributions of UV Ceti-Type M-Dwarf Stars Using TESS Light Curves

## Project Overview

This project studies stellar flares from active M-dwarf stars using high-cadence observations from NASA's Transiting Exoplanet Survey Satellite (TESS).

The project focuses on five nearby UV Ceti-type flare stars:

- EV Lac
- AD Leo
- YZ CMi
- UV Ceti
- Wolf 359

UV Ceti-type stars are active, low-mass M-dwarf stars known for producing short-duration and energetic stellar flares. The project investigates how frequently flares occur and how their estimated energies are distributed.

## Research Question

How can automated wavelet-based analysis of high-cadence TESS light curves be used to detect stellar flares and characterize their energy distributions in UV Ceti-type M-dwarf stars?

## Objectives

1. Obtain high-cadence TESS light curves for selected UV Ceti-type stars.
2. Clean and detrend the observed light curves.
3. Detect possible flare events using wavelet analysis.
4. Validate candidate flare events using their amplitude, duration, morphology, and local noise level.
5. Calculate the equivalent duration (ED) of detected flares.
6. Estimate approximate bolometric-equivalent flare energies.
7. Construct cumulative flare frequency-energy distributions.
8. Fit power-law relationships to the detected flare energies.
9. Compare the flare statistics of the five selected stars.

## Methodology

The analysis pipeline follows this sequence:

TESS Light Curves  
↓  
Data Cleaning and Normalization  
↓  
Savitzky-Golay Detrending  
↓  
Continuous Wavelet Transform  
↓  
Candidate Flare Detection  
↓  
Flare Validation  
↓  
Equivalent Duration  
↓  
Approximate Flare Energy  
↓  
Cumulative Frequency-Energy Distribution  
↓  
Power-Law Analysis

### Flare Detection

A Mexican-hat (`mexh`) wavelet was applied at multiple scales to identify short-duration positive variations associated with possible flares.

Candidate events were selected using an adaptive threshold based on the median absolute deviation (MAD).

Additional validation criteria were applied using flare amplitude, local noise, duration, and light-curve morphology.

### Flare Energy

The equivalent duration of each flare was calculated by integrating the normalized flare excess above the estimated baseline.

Approximate bolometric-equivalent flare energy was then estimated using:

E_flare = ED × L_bol

where the stellar bolometric luminosity was estimated from stellar radius and effective temperature.

These energies should be treated as approximate bolometric-equivalent energies rather than exact TESS-band flare energies.

## Software and Tools

- Python
- Lightkurve
- Astropy
- NumPy
- SciPy
- PyWavelets
- Pandas
- Matplotlib

The project does **not** use AltaiPony in the final implementation.

## Target Stars

| Star | Validated Flares | Used for Power Law | Alpha | Cumulative Slope | R² |
|---|---:|---:|---:|---:|---:|
| EV Lac | 19 | 14 | 1.8613 | -0.7703 | 0.9379 |
| AD Leo | 56 | 42 | 3.3584 | -2.4105 | 0.9597 |
| YZ CMi | 29 | 22 | 1.9527 | -1.1951 | 0.9006 |
| UV Ceti | 39 | 29 | 1.6192 | -0.7341 | 0.9553 |
| Wolf 359 | 12 | 9 | 2.5537 | -1.1318 | 0.8780 |

**Total validated flares: 155**

## Results

The automated pipeline identified 155 validated flare events across the five target stars.

The resulting cumulative frequency-energy distributions were fitted using power-law relationships. The calculated values of the differential power-law index (`alpha`) and cumulative slope are provided in the result CSV files.

The repository also contains the generated detection, validated-flare, power-law, and comparison figures.

## Repository Contents

```text
stellar-flare-statistics/
│
├── stellar_flares.py
├── sf.py
│
├── UV_Ceti_TESS_Flare_Catalog.csv
├── UV_Ceti_Star_Summary.csv
├── UV_Ceti_Power_Law_Results.csv
│
└── images/
    ├── EV_Lac_wavelet_detection.png
    ├── EV_Lac_validated_flares.png
    ├── EV_Lac_power_law.png
    ├── AD_Leo_wavelet_detection.png
    ├── AD_Leo_validated_flares.png
    ├── AD_Leo_power_law.png
    ├── YZ_CMi_wavelet_detection.png
    ├── YZ_CMi_validated_flares.png
    ├── YZ_CMi_power_law.png
    ├── UV_Ceti_wavelet_detection.png
    ├── UV_Ceti_validated_flares.png
    ├── UV_Ceti_power_law.png
    ├── Wolf_359_wavelet_detection.png
    ├── Wolf_359_validated_flares.png
    ├── Wolf_359_power_law.png
    └── All_Stars_Power_Law_Comparison.png
