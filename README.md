# Chirp

**Real-time sound analysis and recording tool for researchers.**

Chirp is a desktop application for multi-stream audio monitoring, visualization, and threshold-triggered recording. It was designed with bioacoustics research in mind but works for any audio analysis task.

![Version](https://img.shields.io/badge/Version-v2.2.0-orange) ![Python](https://img.shields.io/badge/Python-3.11+-blue) ![PyQt5](https://img.shields.io/badge/GUI-PyQt5-green) ![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Features

### Multi-Stream Recording
- Monitor and record from **multiple audio devices simultaneously**
- Independent configuration per stream (device, sample rate, threshold, filters)
- Start/stop acquisition and recording individually or all at once
- Sidebar with live status indicators and mini-amplitude previews

### Real-Time Visualization
- **Spectrogram** display with configurable FFT size, window function, and frequency scale (Linear / Log / Mel)
- **Amplitude envelope** with scrollable Y-axis zoom
- **Raw waveform** display showing signed audio samples (teal color)
- Adjustable display buffer duration (5s – 60s)
- Gain, dB floor, and dB ceiling controls for spectrogram contrast
- Configurable frequency display range
- Scroll-wheel zoom on waveform and spectrogram axes (centered on mouse position)

### Threshold-Triggered Recording
- Automatic recording triggered when amplitude crosses a configurable threshold
- **Pre-trigger buffer** captures audio before the trigger event
- **Hold** bridges gaps between threshold crossings — silent tails are trimmed if no re-crossing occurs
- **Post-trigger window** extends saved audio by a configurable duration after the last crossing
- Adjustable minimum crossing duration and maximum recording duration
- Drag the threshold line directly on the amplitude plot
- **Stereo channel selection** — trigger on Left Channel, Right Channel, Average, Any Channel, or Both Channels
- **Auto-Calibrate** — automatically measure ambient noise and set threshold with configurable calibration duration and margin multiplier
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

### Dropped-Callback Tracking
- `D` badge in the sidebar flashes on each dropped audio callback (queue full)
- Persistent `D` badge latches for the session; click to clear
- In View Mode a per-tile `DROP×N` overlay shows the running session total

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
- **View Mode** — distraction-free monitoring of all streams with adjustable grid columns and panel height

### Theme
- **Catppuccin Mocha** dark theme with teal and peach accent colors

### Settings Persistence
- **Save** and **Save As** for configuration files (`.json` format)
- **Load** restores complete configurations (also reads legacy `.chirp` files)
- All parameters preserved including device names, sample rates, trigger settings, and display options

### Sync Controls
- Optionally synchronize threshold, spectrogram settings, frequency range, and sample rate across all streams

### Audio Monitor Loopback
- Per-stream monitor toggle routes raw audio to a shared output device for live listening

### WAV File Replay (Testing)
- Swap the live capture for a WAV file (`WavFileCapture`) to feed a reproducible signal through the full pipeline — trigger, writer, spectrogram, entropy — for regression testing and offline analysis

---

## What's New in v2.2.0

A robustness-focused release: every issue tagged `data-loss` / `bug` / `robustness` from the v2.1.x audit (#43–#58) is now closed. No new user-facing features; the focus is on never silently losing a recording.

- **Error surfacing** — sticky red badge in the sidebar latches on the first writer-pool / ingest-thread failure with a tooltip and reset-on-click (#44, #43, #48).
- **Safe teardown flush** — Stop Acq, change device, change sample rate, switch-to-WAV, and remove-stream all now flush in-flight trigger events through `_stop_ingest_and_flush` so a tone in progress is never discarded (#45).
- **Atomic WAV writes** — every WAV is written to a sibling `.tmp` file, fsynced, then `os.replace`-d, so a crash mid-write leaves the old file untouched or the new file complete — never a truncated header (#52).
- **Writer-pool resilience** — workers are supervised and respawn on death; queue-backlog watermark + respawn count exposed for telemetry (#47).
- **Graceful close** — `closeEvent` drains the writer pool with a modal progress dialog and surfaces partial-failure summaries (#56).
- **DSP lock + ingest thread** — buffer reallocation across sample-rate changes is locked against concurrent ingest (#53).
- **Sample-rate hardening** — events flushed mid-SR-change carry the original capture rate in the WAV header (#46).
- **Filename + path hygiene** — Windows reserved names, path traversal, length caps, blank `output_dir`, and realpath containment all guarded at the writer entry point (#51, #50).
- **Schema-validated config loader** — `_load_settings` routes through the schema validator and bails BEFORE teardown on a bad file, surfacing warnings in a modal (#55).
- **WAV replay correctness** — missing-file no longer falls back to the live mic; multi-channel WAV truncation is surfaced in the sidebar (#49, #54).
- **Display-thread guard** — `_update_plot` body wrapped in a top-level try/except; on persistent failure a sticky "DISPLAY HALTED — acquisition still running" note appears so the user knows NOT to force-kill (#58).
- **`max_rec` butt-joined continuation** — a force-split now opens a continuation event immediately at the boundary (no `min_cross` re-qualification gate). The two halves butt-join sample-accurately and carry `_part01` / `_part02` filename suffixes so the WAV series is unambiguous (#57).
- **Default folder auto-create** — the default `./recordings` folder is created on first run instead of being flagged as an invalid path.

## What's New in v2.1.0

- **Sticky health badges (#28, #29)** — sidebar gets session-wide `S` (saturation) and `D` (drops) indicators that latch on the first event and stay lit until clicked, so brief clips and single dropped audio callbacks never slip by. Mirrored as `SAT` / `DROP×N` overlays in View Mode so the monitoring grid surfaces the same flags.
- **Detect / record events strip (#32)** — a two-row strip under the amplitude plot shows, sample by sample, when the trigger fired (yellow) and when audio was captured (green). Shared with the state machine (one source of truth, respects bandpass filter + spectral entropy gating). Visible in both Config and View modes.
- **Audio monitor loopback (#7)** — per-stream toggle sends raw audio to a shared output device.
- **WAV file replay (#27)** — swap the live input for a WAV file to feed reproducible signals through the full pipeline (useful for regression tests and offline analysis).
- **Compact UI** — config panels merged into a single row and sliders removed, freeing canvas space.
- **Robustness** — dropped-callback counter per capture, per-entity ingestion thread to decouple DSP from the PortAudio callback.

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
- `scipy` — signal processing and WAV output
- `matplotlib` — spectrogram rendering
- `PyQt5` — GUI framework

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
