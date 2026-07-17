# Chirp

**Real-time sound analysis and recording tool for researchers.**

Chirp is a desktop application for multi-stream audio monitoring, visualization, and threshold-triggered recording. It was designed with bioacoustics research in mind but works for any audio analysis task.

![Version](https://img.shields.io/badge/Version-v3.5.1-orange) ![Python](https://img.shields.io/badge/Python-3.11+-blue) ![PyQt5](https://img.shields.io/badge/GUI-PyQt5%20%2B%20pyqtgraph-green) ![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Features

### Multi-Stream Recording
- Monitor and record from **multiple audio devices simultaneously**
- Independent configuration per stream (device, sample rate, threshold, filters)
- Start/stop acquisition and recording individually or all at once
- Sidebar with live status indicators and mini-amplitude previews

### Real-Time Visualization
- **Efficient real-time rendering** via pyqtgraph with an OpenGL-composited viewport (both Config and View modes) — cheap per-frame image blits (no full-figure re-raster), auto-downsampled curves, off-screen tile culling, and an adaptive frame rate keep many streams cheap
- **Spectrogram** display with configurable FFT size, window function, and frequency scale (Linear / Log / Mel)
- **Amplitude envelope** with mouse-wheel Y-axis zoom and per-stream **linear / log (dB) Y scale** (right-click the plot to switch; defaults to log)
- **Raw waveform** display showing signed audio samples (teal color)
- Adjustable display buffer duration (5s – 60s)
- Gain, dB floor, and dB ceiling controls for spectrogram contrast
- Configurable frequency display range
- Mouse-wheel Y-zoom on the amplitude, waveform, and entropy plots

### Threshold-Triggered Recording
- Automatic recording triggered when amplitude crosses a configurable threshold
- **Pre-trigger buffer** captures audio before the trigger event
- **Hold** bridges gaps between threshold crossings — silent tails are trimmed if no re-crossing occurs
- **Post-trigger window** extends saved audio by a configurable duration after the last crossing
- Adjustable minimum crossing duration and maximum recording duration
- Drag the threshold line directly on the amplitude plot
- **Stereo channel selection** — trigger on Left Channel, Right Channel, Average, Any Channel, or Both Channels
- **Auto-Calibrate** — automatically measure ambient noise and set threshold with configurable calibration duration and margin multiplier
- **Force Trigger toggle** — press to open a recording segment immediately (pre-trigger lookback included), press again to close it on the spot; Max Rec splitting still applies
- **Continuous recording mode** (per stream) — record everything while REC is on, rotating to a new file every Max Rec seconds; all other trigger parameters are ignored
- **Entropy Min Duration** — a debounce for the spectral gate: entropy must stay below its threshold continuously for the configured time before the spectral condition turns ON
- **Detect / record events strip** beneath the amplitude plot — a yellow row lights up for every sample the trigger condition is met, a green row for every sample being written to the WAV file. Derived from the same per-sample mask the state machine consumes (no parallel computation), so the strip is exact — including bandpass and spectral-entropy gating. Visible in both Config and View modes.

### Spectral Entropy Trigger
- Detect tonal sounds by monitoring spectral entropy computed from FFT magnitudes
- Normalized Shannon entropy: 0 = pure tone, 1 = white noise
- **Four detection modes**: Amplitude Only (default), Spectral Only, Amp AND Spectral, Amp OR Spectral
- Configurable entropy threshold (triggers when entropy falls **below** the threshold)
- Real-time entropy trace plot (appears when a spectral mode is active)
- Draggable entropy threshold line on the plot
- Live entropy readout in the status panel
- Shares the same FFT window as the spectrogram for efficiency

### Saturation Detection
- Real-time clipping detection when audio peaks reach ≥ 99% of full scale
- Amplitude waveform turns **red** to alert on saturation
- **Sticky `S` badge** in the sidebar latches on the first clip and stays lit until clicked — brief clips are never missed
- In View Mode the per-tile `SAT` overlay surfaces the same state across the monitoring grid

### Lossless Capture & Drop Tracking
- Raw audio lands in a preallocated per-stream **ring buffer** straight from the realtime callback (a lock-free memcpy — no allocation, no disk I/O, no logging on the audio thread), so capture cannot be stalled by DSP or rendering
- `D` badge in the sidebar flashes if the ring ever overruns (the DSP consumer fell seconds behind — should never happen in normal use)
- Persistent `D` badge latches for the session; click to clear
- In View Mode a per-tile `DROP×N` overlay shows the running session total

### Trustworthy Timestamps
- Filename timestamps come from a **disciplined capture clock**: sample-accurate relative timing (adjacent and Max-Rec–split files tile exactly), continuously steered onto the system wall clock so sound-card crystal drift never accumulates over multi-week runs
- Internally UTC, rendered to local time only in the filename — DST transitions mid-run don't skew timestamps
- Capture holes (device stall, drop burst) are jumped **between** recordings, never inside a file, and logged
- **Publish-time watchdog** — every finalized WAV's onset + duration is checked against the wall clock; divergence beyond 10 s lights the sidebar `!` badge and is logged with the exact delta
- **Clock audit log** (`chirp_clock_log.csv`, one row/min/stream) pairs the capture sample index with the derived time and the raw system clock, so any filename timestamp can be verified — or corrected — offline after a long experiment

### Error Logging
- Every pipeline failure surfaced by the sidebar `S` / `D` / `!` indicators is also written to a plain-text log file (`chirp_errors.log`) in the folder Chirp is launched from
- Each line carries an ISO timestamp, the category, the stream name, and (where applicable) the WAV file path involved — so any indicator can be traced back to a precise event
- Categories: `ring_overrun` (capture ring overrun), `os_drop` (PortAudio overflow), `ingest` (DSP-thread exceptions, with a short traceback), `open` (capture / WAV open failure), `wav_writer` (writer-pool failure, with target folder), `saturation` (one line per finished WAV that contained clipping, with the full path), `clock_step` (timestamp clock jumped a capture hole), `timestamp_divergence` (published WAV's timestamp disagrees with the wall clock, with the full path)
- All disk logging happens on a dedicated background thread — realtime and DSP threads only enqueue a record, so logging can never stall the audio pipeline
- High-frequency categories (`ring_overrun`, `os_drop`) are throttled to one entry per stream per second; cumulative counts are stamped on each line
- Saturation is logged **per file**, not per sample — one line per WAV that clipped — so the log stays compact even on noisy inputs
- The log is append-only (survives across runs) and the logger never raises — it cannot crash the audio pipeline

### Bandpass Filter
- Optional Butterworth bandpass filter per stream
- Configurable low and high frequency cutoffs

### Flexible Output
- Custom output folder, filename prefix, and suffix per stream
- **Reference date tracking** with automatic day-count subfolder naming (e.g., for days post-hatch)

### Three Visualization Modes
- Selectable via the **View** combo in the Display panel:
  - **Spectrogram** (default) — spectrogram + amplitude envelope
  - **Waveform** — raw signed audio waveform (teal) + amplitude envelope
  - **Both** — spectrogram + waveform + amplitude envelope
- Works in stereo and in View Mode

### Two Display Modes
- **Config Mode** — full control panel for adjusting all parameters
- **View Mode** — distraction-free monitoring of all streams in a grid, with adjustable **columns**, per-tile **height**, and a **Fit to screen** button that sizes the tiles so every stream is visible without scrolling. Each tile carries a header with the stream name, **acq/rec/trig status dots**, and the sticky **S / D / !** badges (click to clear) — plus the detect/record **events strip** under the amplitude plot. An **"Active only"** toggle hides idle streams. Off-screen tiles are skipped while rendering, and the frame rate adapts under load so audio capture always takes priority.

### Theme
- **Catppuccin Mocha** dark theme with teal and peach accent colors

### Settings Persistence
- **Save** and **Save As** for configuration files (`.json` format)
- **Load** restores complete configurations (also reads legacy `.chirp` files)
- All parameters preserved including device names, sample rates, trigger settings, and display options

### Bulk Editing
- **Apply All Settings** — one-shot copy of every setting from the selected stream to all others
- **All-Streams Table** — an editable side-by-side table of every parameter of every stream (double-click a cell to edit; structural params like device and sample rate are shown read-only)

### Audio Monitor Loopback
- Per-stream monitor routes raw audio to a shared output device for live listening
- **Enable/disable toggle** that remembers the source and output selections — unmute restores the exact same routing
- **Follow-selection mode** — the monitor automatically re-targets to the stream selected in the sidebar (Config mode) or the tile clicked in the grid (View mode); the monitored tile shows a 🎧 marker

### WAV File Replay (Testing)
- Swap the live capture for a WAV file (`WavFileCapture`) to feed a reproducible signal through the full pipeline — trigger, writer, spectrogram, entropy — for regression testing and offline analysis

---

## Release Notes

Per-version release notes (what's new in each version) live on the [GitHub Releases page](https://github.com/TamirEliav/Chirp/releases).

---

## Installation

### Requirements
- Python 3.11+
- A working audio input device

### Setup

```bash
# Clone the repository
git clone https://github.com/TamirEliav/Chirp.git
cd Chirp

# Create a conda environment (recommended)
conda create -n chirp python=3.11
conda activate chirp

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
- `sounddevice` — audio capture
- `numpy` — numerical processing
- `scipy` — signal processing (`scipy.fft`, filtering)
- `soundfile` — streaming WAV output (libsndfile)
- `PyQt5` — GUI framework
- `pyqtgraph` + `PyOpenGL` — GPU-accelerated real-time plotting
- `matplotlib` — colormap source (and a small legacy helper layer)

---

## Usage

```bash
python chirp.py
```

### Quick Start
1. Launch the app — it opens maximized with one recording stream
2. Select an audio input device from the dropdown
3. Click **Start Acq** to begin monitoring
4. Adjust the threshold (drag the line on the amplitude plot or use the slider)
5. Click **Start Rec** to enable threshold-triggered recording
6. Set an output folder for saved WAV files

### Adding Multiple Streams
- Click **Add Recording** to create additional streams
- Each stream can use a different device and settings
- Switch to **View Mode** for a clean multi-stream monitoring layout

### Saving Your Setup
- Use **Save Settings** to export your full configuration to a `.json` file
- Use **Load Settings** to restore it later (also reads legacy `.chirp` files)

---

## User Manual

A detailed HTML manual covering all features, settings, and workflows is included in this repository.

**[Open User Manual](https://htmlpreview.github.io/?https://github.com/TamirEliav/Chirp/blob/master/manual.html)**

---

## Supported Sample Rates

8000 · 16000 · 22050 · 44100 · 48000 · 96000 Hz

---

## License

MIT
