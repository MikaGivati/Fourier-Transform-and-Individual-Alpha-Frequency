
# Individual Alpha Frequency from EEG Power Spectra

Extracting each subject's **Individual Alpha Frequency (IAF)** from resting-state EEG,
by comparing eyes-open vs. eyes-closed power spectra computed with five different
spectral estimation methods.

The alpha band (8–12 Hz) typically shows a power "bump" when a subject's eyes are
closed, but the exact peak frequency varies from person to person. This project
computes that peak — the IAF — for 3 subjects, using 5 independent implementations of
power spectral density (PSD) estimation, and compares both their results and their
runtime.

## What's here

- **Automated file discovery** — subject folders are scanned and matched by filename
  pattern (regex), rather than hardcoded paths, since raw filenames were inconsistent.
- **Custom EDF reader** (`edf.py`) — parses the EEG files directly from the European
  Data Format spec (header + signal records), no external EEG library.
- **Five PSD estimation methods**, applied to the Pz electrode (channel 19) of each
  recording:
  1. **FFT** — direct `scipy.fft` power spectrum
  2. **Welch's method** — `scipy.signal.welch` (windowed & averaged)
  3. **Custom Welch (FFT-based)** — windowing + FFT implemented from scratch with
     `numpy.lib.stride_tricks.sliding_window_view`, fully vectorized (no loops)
  4. **Custom Welch (DFT-based)** — same, but the transform itself is computed via an
     explicit Vandermonde matrix instead of calling FFT
  5. **Bonus: full-signal DFT** — the whole-signal transform via a chunked Vandermonde
     matrix multiplication (to keep memory bounded), equivalent to FFT but computed
     the "slow" way
- **IAF extraction** — for each method, subtracts the eyes-open spectrum from the
  eyes-closed spectrum and finds the frequency of maximum power within the alpha band.
- **Runtime comparison** — benchmarks all 5 methods across all subjects/tasks.

## Results

**Power spectra comparison** (Subject 3, eyes-open vs. eyes-closed, all 5 methods):

![Power spectra comparison](figures/spectra_subject3.png)

**IAF extraction** (Subject 2) — the vertical line marks the frequency of maximum
power difference within the alpha band:

![IAF calculation](figures/iaf_subject2.png)

**IAF across all subjects and methods:**

![IAF table](figures/iaf_table.png)

The IAF estimate is highly consistent across methods for the same subject (methods
agree to within ~0.1 Hz), while the differences *between* subjects are much larger —
i.e. IAF is a stable, individual trait, not an artifact of which estimator you use.

**Runtime comparison across methods** (log scale):

![Runtime comparison](figures/runtime_barplot.png)

The naive full-signal DFT (bonus method) is ~4 orders of magnitude slower than FFT-based
approaches, since it explicitly builds and multiplies an N×N Vandermonde matrix instead
of using a fast algorithm — a nice illustration of why FFT exists.

## Repo structure

```
├── src/
│   ├── main.py          # full analysis pipeline
│   ├── edf.py            # EDF file reader
│   └── args_parser.py    # CLI args (window size, overlap, electrode, data dir)
├── data/
│   ├── subject1/          # eyes-open + eyes-closed .edf recordings
│   ├── subject2/
│   └── subject3/
├── figures/               # figures used in this README
└── requirements.txt
```

## Running it

```bash
pip install -r requirements.txt
cd src
python main.py
```

Optional CLI arguments (see `args_parser.py`):

| Flag | Default | Description |
|---|---|---|
| `--elec_num` | 18 | Electrode index to analyze (18 = Pz, 0-indexed) |
| `--DATA_DIR` | `data` | Path to the data folder |
| `--window_size` | 4 | Welch window size, in seconds |
| `--overlap` | 2 | Welch window overlap, in seconds |

Note: the bonus full-signal DFT method is intentionally slow (~30s per recording) —
it's included to demonstrate the runtime cost of not using FFT, not for speed.

## Notes

- This was a group assignment.
- Data loading is fully automated: filenames don't follow a fixed convention (some
  have extra spaces, inconsistent ordering), so subject number and task (EO/EC) are
  extracted via regex rather than hardcoded.
- All windowing and transform operations are vectorized — no explicit loops over time
  windows.
