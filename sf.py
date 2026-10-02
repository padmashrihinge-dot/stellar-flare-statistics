import sys
import numpy.linalg

# Monkey-patch for Python 3.13 + AltaiPony / k2sc compatibility
sys.modules['numpy.linalg.linalg'] = numpy.linalg

import lightkurve as lk
import altaipony as altai
from altaipony.flarelc import FlareLightCurve
import matplotlib.pyplot as plt
import numpy as np

# 1. Fetch TESS light curve via lightkurve
print("Searching MAST for TESS observations of EV Lac...")
search_result = lk.search_lightcurve("EV Lac", mission="TESS")
lc = search_result[0].download().remove_nans()

# 2. Convert to AltaiPony FlareLightCurve object
flc = FlareLightCurve(time=lc.time.value, flux=lc.flux.value, flux_err=lc.flux_err.value)

# 3. Detrend baseline stellar variability using Savitzky-Golay filter
flc = flc.detrend("savgol")

# 4. Find flares automatically
flc = flc.find_flares()

# Inspect detected flare columns in pandas DataFrame
print("\n--- Detected Flares Summary ---")
print(f"Total flares found: {len(flc.flares)}")

# AltaiPony uses 'tstart', 'tstop', 'ed_rec' or 'ed_savgol'
columns_to_show = [col for col in ['tstart', 'tstop', 'ed_rec', 'ed_savgol', 'total_flux'] if col in flc.flares.columns]
print(flc.flares[columns_to_show])

# -------------------------------------------------------------
# 5. CUMULATIVE FREQUENCY-ENERGY POWER LAW PLOT
# -------------------------------------------------------------
# Select energy column (ed_rec if recovery step run, otherwise ed_savgol)
ed_col = 'ed_rec' if 'ed_rec' in flc.flares.columns else 'ed_savgol'

if ed_col in flc.flares.columns and len(flc.flares) > 0:
    energies = flc.flares[ed_col].dropna().values
    sorted_energies = np.sort(energies)[::-1]
    cumulative_counts = np.arange(1, len(sorted_energies) + 1)

    plt.figure(figsize=(8, 5))
    plt.loglog(sorted_energies, cumulative_counts, 'ro-', lw=1.5, markersize=5)
    plt.title('Cumulative Flare Energy Distribution (EV Lacertae)')
    plt.xlabel('Equivalent Duration [Seconds/Days]')
    plt.ylabel('Cumulative Frequency N(>E)')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.show()