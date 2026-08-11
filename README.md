# Chirp

**Real-time sound analysis and recording tool for researchers.**

Chirp is a desktop application for multi-stream audio monitoring, visualization, and threshold-triggered recording. It was designed with bioacoustics research in mind but works for any audio analysis task.

![Version](https://img.shields.io/badge/Version-v3.9.0-orange) ![Python](https://img.shields.io/badge/Python-3.11+-blue) ![PyQt5](https://img.shields.io/badge/GUI-PyQt5%20%2B%20pyqtgraph-green) ![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Table of Contents

- [Features](#features)
- [Release Notes](#release-notes)
- [Installation](#installation)
  - [Requirements](#requirements)
  - [Step 1 — Install Miniforge](#step-1--install-miniforge)
  - [Step 2 — Get Chirp and create its environment](#step-2--get-chirp-and-create-its-environment)
  - [Dependencies](#dependencies)
- [Usage](#usage)
  - [Quick Start](#quick-start)
  - [Adding Multiple Streams](#adding-multiple-streams)
  - [Saving Your Setup](#saving-your-setup)
- [User Manual](#user-manual)
- [Supported Sample Rates](#supported-sample-rates)
- [License](#license)

---

## Features

### Multi-Stream Recording
- Monitor and record from **multiple audio devices simultaneously**
- Independent configuration per stream (device, sample rate, threshold, filters)
- Start/stop acquisition and recording individually or all at once
- Sidebar with live status indicators and mini-amplitude previews
- **Parameter lock** (per stream, or Lock All / Unlock All) — a 🔒 toggle freezes a stream's configuration against accidental edits; display params, the audio monitor, and start/stop still work, and unlocking asks for confirmation naming the stream so one user can't unlock another's by mistake
- **Adjustable layout** — drag the borders between the sidebar and the main pane, and between the display plots and the configuration panels

### Real-Time Visualization
- **Efficient real-time rendering** via pyqtgraph with an OpenGL-composited viewport (both Config and View modes) — cheap per-frame image blits (no full-figure re-raster), auto-downsampled curves, off-screen tile culling, and an adaptive frame rate keep many streams cheap
- **Spectrogram** display with configurable FFT size, window function, and frequency scale (Linear / Log / Mel)
- **Amplitude envelope** with mouse-wheel Y-axis zoom and per-stream **linear / log (dB) Y scale** (right-click the plot to switch; defaults to log)
- **Raw waveform** display showing signed audio samples (teal color)
- Adjustable display buffer duration (5s – 60s)
- Gain, dB floor, and dB ceiling controls for spectrogram contrast
- Configurable frequency display range
- Mouse-wheel Y-zoom on the amplitude, waveform, and entropy plots
- **Smooth, audio-aligned scrolling** — the view advances at wall-clock rate rather than as audio is delivered, so it scrolls smoothly even on capture backends that hand over a whole device buffer at a time (WASAPI exclusive, WDM-KS). The red cursor is steered onto the sample the **audio monitor is currently playing** — the monitor's jitter buffer *and* the output device's latency are both accounted for — so what you see and what you hear line up

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

### Capture Engine (⚙ Advanced)
- **Input buffer (latency)** and **callback block size** are tunable per machine — the settings that decide how much slack the driver has before captured audio is lost, and how often Chirp is woken to collect it. Neither affects the recorded audio, only display/monitor delay
- **WASAPI exclusive mode** — hands the input endpoint to Chirp alone so capture comes straight from the driver instead of through the shared Windows audio engine (the layer implicated in inserted silence). WASAPI devices only; on MME / DirectSound / WDM-KS the request is logged and ignored, and a refused exclusive open falls back to a shared open rather than leaving an unattended rig with no capture. The `open` log line names the device, its host API, and whether exclusive mode actually took
- **Inserted-silence auto-recovery** — when a stream's digital-zero duty cycle stays above a configurable threshold for a configurable time, Chirp restarts acquisition on **every** stream sharing that device (the only reset known to clear a latched capture session) and resumes recording, with a cooldown to prevent restart loops. Every intervention is logged

### Inserted-Silence Detection
- Detects **digital silence injected below the app** — a driver or the Windows audio engine can zero-fill milliseconds of a capture in place, corrupting the spectrogram, the monitor and every recorded WAV, sometimes *without raising any PortAudio error flag*
- Two independent detectors: PortAudio's `input_underflow` status flag (zero samples inserted to cover missing data), and a **signal-level scan** for exact-zero runs ≥ 1 ms — a live analog input's noise floor never produces those, so such a run is inserted silence by definition
- Lights the sticky `!` badge with the run count and longest gap, and logs `underflow` / `zero_run` lines so a corrupted session is visible **while it happens** instead of being discovered in the recordings afterwards
- The first second after Start Acq is exempt (a priming capture pin legitimately delivers silence)
- Chirp also warns at stream open when the device's default format doesn't match the stream's sample rate — hidden OS resampling that can mask discontinuities

### One Capture Stream per Device
- All streams that share an input device now share **one** PortAudio stream, so two streams splitting a stereo input (e.g. Left on one, Right on the other) present a **single client** to the operating system's capture session rather than two
- Each stream still keeps its own ring buffer, clock, trigger state, monitor routing and error counters — the buffer is simply fanned out
- The device is released when the **last** stream on it stops, which is what resets a driver/engine capture session that has latched into a bad state
- Note: two streams on the same device at *different* sample rates cannot share and will still open two sessions

### Trustworthy Timestamps
- Filename timestamps come from a **disciplined capture clock**: sample-accurate relative timing (adjacent and Max-Rec–split files tile exactly), continuously steered onto the system wall clock so sound-card crystal drift never accumulates over multi-week runs
- Internally UTC, rendered to local time only in the filename — DST transitions mid-run don't skew timestamps
- Capture holes (device stall, drop burst) are jumped **between** recordings, never inside a file, and logged
- **Publish-time watchdog** — every finalized WAV's onset + duration is checked against the wall clock; divergence beyond 10 s lights the sidebar `!` badge and is logged with the exact delta
- **Clock audit log** (`chirp_clock_log.csv`, one row/min/stream) pairs the capture sample index with the derived time and the raw system clock, so any filename timestamp can be verified — or corrected — offline after a long experiment
- `scripts/check_clock_log.py` audits that log for **lost** samples — audio the driver never delivered, which no in-app counter can see: it compares how fast the sample index advances against the wall clock, where a healthy capture agrees to within crystal difference (tens of ppm)

### Error Logging
- Every pipeline failure surfaced by the sidebar `S` / `D` / `!` indicators is also written to a plain-text log file (`chirp_errors.log`) in the folder Chirp is launched from
- Each line carries an ISO timestamp, the category, the stream name, and (where applicable) the WAV file path involved — so any indicator can be traced back to a precise event
- Categories: `ring_overrun` (capture ring overrun), `os_drop` (PortAudio overflow), `underflow` (PortAudio inserted zero samples into the captured audio), `zero_run` (exact-zero runs found in the captured signal — inserted silence with no PortAudio flag), `ingest` (DSP-thread exceptions, with a short traceback), `open` (capture / WAV open failure), `wav_writer` (writer-pool failure, with target folder), `saturation` (one line per finished WAV that contained clipping, with the full path), `clock_step` (timestamp clock jumped a capture hole), `timestamp_divergence` (published WAV's timestamp disagrees with the wall clock, with the full path)
- All disk logging happens on a dedicated background thread — realtime and DSP threads only enqueue a record, so logging can never stall the audio pipeline
- High-frequency categories (`ring_overrun`, `os_drop`, `underflow`, `zero_run`) are throttled to one entry per stream per second; cumulative counts are stamped on each line
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
- **All-Streams Table** — an editable side-by-side table of every parameter of every stream (double-click a cell to edit; structural params like device and sample rate are shown read-only)

### Audio Monitor Loopback
- Per-stream monitor routes raw audio to a shared output device for live listening
- **Enable/disable toggle** that remembers the source and output selections — unmute restores the exact same routing
- **Follow-selection mode** — the monitor automatically re-targets to the stream selected in the sidebar (Config mode) or the tile clicked in the grid (View mode); the monitored tile shows a 🎧 marker
- **Self-tuning jitter buffer** — capture backends that deliver one device buffer at a time (WASAPI exclusive, WDM-KS) hand over half a second of audio at once and then nothing; the monitor holds back a small reserve and grows it automatically until playback is gap-free, so bursty capture doesn't stutter

### WAV File Replay (Testing)
- Swap the live capture for a WAV file (`WavFileCapture`) to feed a reproducible signal through the full pipeline — trigger, writer, spectrogram, entropy — for regression testing and offline analysis

---

## Release Notes

Per-version release notes (what's new in each version) live on the [GitHub Releases page](https://github.com/TamirEliav/Chirp/releases).

---

## Installation

### Requirements
- Python 3.11+ (installed for you by Miniforge in Step 1 below)
- A working audio input device

> **New to Python?** You don't need to install Python yourself or understand how
> environments work. Just follow the two steps below in order and copy-paste the
> commands — the whole setup takes a few minutes.

### Step 1 — Install Miniforge

**Miniforge** is a small, free, community-maintained installer for Python and
`conda` (it uses the open `conda-forge` package repository by default). `conda`
creates an isolated "environment" — a self-contained copy of Python and Chirp's
dependencies that won't interfere with anything else on your computer. This is
the recommended way to run Chirp because it keeps the exact Python version Chirp
expects separate from your system.

1. Download the installer for your operating system from the official releases
   page: **[Miniforge on GitHub](https://github.com/conda-forge/miniforge#miniforge3)**
   (pick the **Miniforge3** installer that matches your OS and CPU).
2. Run the installer and accept the defaults.
   - **Windows:** after it finishes, open the **"Miniforge Prompt"** from the Start
     menu — this is the terminal where the commands below will work.
   - **macOS / Linux:** open a new Terminal window.

A more detailed walkthrough is in Miniforge's own
[installation instructions](https://github.com/conda-forge/miniforge#install).

### Step 2 — Get Chirp and create its environment

Run these commands one at a time in the Miniforge Prompt (Windows) or Terminal
(macOS / Linux):

```bash
# Clone the repository (or download it as a ZIP from GitHub and unzip it)
git clone https://github.com/TamirEliav/Chirp.git
cd Chirp

# Create an isolated environment named "chirp" with the right Python version
conda create -n chirp python=3.11

# Activate it — do this every time before running Chirp
conda activate chirp

# Install Chirp's dependencies into the environment
pip install -r requirements.txt
```

Setup is a one-time step. From now on you only need `conda activate chirp` before
launching the app (see [Usage](#usage)).

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
# Activate the environment first (once per terminal session)
conda activate chirp

# Launch Chirp
python -m chirp
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
