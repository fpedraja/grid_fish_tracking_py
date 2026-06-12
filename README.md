# Grid Fish Tracking — Python Port

Python + PyQt5 port of the MATLAB electric fish tracking system ([AUD_teensy9.m](https://github.com/fpedraja/Grid_fish_tracking)).

Detects and tracks weakly electric fish (EOD signals) from multi-channel WAV recordings produced by the TeeGrid R40 logger (8 channels, 20 kHz).

## Features

- Optional 50/60 Hz comb notch filter to remove powerline interference before bandpass
- Bandpass filter (300–2000 Hz) + Hilbert envelope peak detection
- Two localization methods: Gaussian spatial grid (argmax) or weighted centroid (Henninger et al. 2020 JEB)
- DBSCAN spatial clustering of EOD events per file
- Kalman filter tracker with Hungarian assignment across files
- PyQt5 GUI with:
  - 2-D tank track map (fish trajectories) with exact grid bounds
  - EOD frequency vs file index plot, colour-coded per fish — below the track map
  - 1-second bandpass signal viewer with colour-coded EOD markers and manual Y-scale lock
  - PCA/UMAP cluster identity scatter plot (amplitude fingerprints)
  - Live log and results table
  - CSV export

---

## Installation

### 1. Install Miniconda (if not already installed)

Download from https://docs.conda.io/en/latest/miniconda.html

### 2. Create the environment

```bash
conda create -n fish_tracker python=3.11 -y
conda activate fish_tracker
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `umap-learn` is optional but recommended for UMAP cluster visualisation.
> It installs numba/llvmlite which are large packages (~500 MB).
> Without it the GUI falls back to PCA automatically.
>
> ```bash
> pip install umap-learn   # optional
> ```

---

## Running the GUI

```bash
conda activate fish_tracker
python main.py
```

---

## Usage

1. Click **Browse…** and select a folder containing 1-minute WAV files from the TeeGrid R40 recorder (8 channels, 20 kHz).
2. Adjust parameters in the left panel if needed (see below).
3. Click **▶ Start** to process all files.
4. Switch to the **Tracks & Signals** tab to see:
   - Fish track map — top left, bounded exactly to the tank grid
   - EOD frequency vs file index — bottom left, same fish colours as the track map
   - 1-second filtered signal with EOD markers — top right
   - PCA/UMAP cluster identity scatter — bottom right
5. Use the **File** slider to scrub back through any processed minute.
6. Click **⬇ Export CSV** to save track data.

---

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `notch_enabled` | off | Enable 50/60 Hz comb notch filter (removes powerline harmonics) |
| `notch_hz` | 50 Hz | Powerline fundamental frequency (50 Hz in Uruguay/Europe, 60 Hz in Americas) |
| `notch_Q` | 30 | Notch quality factor — higher = narrower notch per harmonic |
| `localization_method` | gaussian_grid | `gaussian_grid` or `weighted_centroid` (Henninger et al. 2020) |
| `top_n_electrodes` | 4 | Electrodes used per event in weighted centroid mode |
| `bp_low / bp_high` | 300 / 2000 Hz | Bandpass filter range |
| `min_pk_height` | 0.015 | Envelope amplitude threshold for EOD detection |
| `min_events_for_fish` | 600 | Minimum EOD events per file before clustering |
| `eps_phys` | 20 cm | DBSCAN neighbourhood radius |
| `min_pts` | 60 | DBSCAN minimum cluster size |
| `min_events_per_cluster` | 200 | Minimum events per valid cluster |
| `hard_gate_pos` | 60 cm | Kalman tracker max position jump between files |
| `hard_gate_f` | 2 Hz | Kalman tracker max frequency jump between files |
| `max_miss` | 10 | Files a track can coast without a detection |

### Notch filter

If you see 50/60 Hz powerline interference in the signal viewer, enable the **Notch Filter** in the Parameters panel.
The comb notch removes the fundamental frequency and all its harmonics in one pass before the bandpass filter is applied.
Set `notch_hz` to match your local grid frequency (50 Hz in Uruguay/Europe, 60 Hz in most of the Americas).
Increase `notch_Q` for a narrower notch if you are concerned about attenuating nearby EOD frequencies.

### Localization method

Two methods are available in the **Localization** section of the Parameters panel:

- **Gaussian grid (argmax)** — precomputes a Gaussian spatial kernel over a dense 1 cm grid and picks the grid point with the highest weighted score. Resolution is limited to the grid step. Controlled by `sigma_spatial`.
- **Weighted centroid (Henninger 2020)** — implements the method from [Henninger et al. (2020) JEB](https://journals.biologists.com/jeb/article/223/3/jeb206342/223694): `r̂ = Σ(√Aᵢ · rᵢ) / Σ(√Aᵢ)` using only the `top_n_electrodes` with the largest amplitude per event. Gives sub-electrode-spacing precision and is simpler to interpret. Recommended starting point: `top_n_electrodes = 4`.

### Tuning tips

- If you see many **"No clusters found"** with fish clearly visible in the signal viewer, increase `eps_phys` (try 40–60 cm) or reduce `min_pts` (try 20).
- If spurious low-frequency tracks appear (e.g. 28 Hz), raise `min_freq_hz` (try 20 Hz).
- `min_events_for_fish` can be lowered if recordings are shorter than 60 s.

---

## Sensor geometry

Default layout matches the TeeGrid R40 with 8 hydrophones in a 80 × 120 cm tank:

```
ch1 (0,0)    ch2 (0,80)
ch3 (40,0)   ch4 (40,40)   ch5 (40,80)   ch6 (40,120)
ch7 (80,40)  ch8 (80,120)
```

Edit the **Sensor geometry** section in the Parameters panel to match your setup.

---

## File format

WAV files must be multi-channel (8 ch), 20 kHz, any standard PCM format.
Files with non-standard LIST metadata chunks (common with TeeGrid) are handled automatically via a scipy fallback reader.

---

## Output CSV columns

| Column | Description |
|--------|-------------|
| `file_idx` | File index (1-based) |
| `filename` | WAV filename |
| `track_id` | Unique fish track ID |
| `x_cm` | Estimated X position (cm) |
| `y_cm` | Estimated Y position (cm) |
| `freq_hz` | Estimated EOD frequency (Hz) |
| `std_x_cm` | Position uncertainty X (cm) |
| `std_y_cm` | Position uncertainty Y (cm) |
