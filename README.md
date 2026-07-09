# Chirp

**Real-time sound analysis and recording tool for researchers.**

Chirp is a desktop application for multi-stream audio monitoring, visualization, and threshold-triggered recording. It was designed with bioacoustics research in mind but works for any audio analysis task.

![Version](https://img.shields.io/badge/Version-v3.2.0-orange) ![Python](https://img.shields.io/badge/Python-3.11+-blue) ![PyQt5](https://img.shields.io/badge/GUI-PyQt5%20%2B%20pyqtgraph-green) ![License](https://img.shields.io/badge/License-MIT-yellow)

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

### Error Logging
- Every pipeline failure surfaced by the sidebar `S` / `D` / `!` indicators is also written to a plain-text log file (`chirp_errors.log`) in the folder Chirp is launched from
- Each line carries an ISO timestamp, the category, the stream name, and (where applicable) the WAV file path involved — so any indicator can be traced back to a precise event
- Categories: `ring_overrun` (capture ring overrun), `os_drop` (PortAudio overflow), `ingest` (DSP-thread exceptions, with a short traceback), `open` (capture / WAV open failure), `wav_writer` (writer-pool failure, with target folder), `saturation` (one line per finished WAV that contained clipping, with the full path)
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
- **View Mode** — distraction-free monitoring of all streams in a grid, with adjustable **columns**, per-tile **height**, and a **Fit to screen** button that sizes the tiles so every stream is visible without scrolling. Off-screen tiles are skipped while rendering, and the frame rate adapts under load so audio capture always takes priority.

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

## What's New in v3.2.0

The Phase-C cleanup release completing the v3.0 → v3.1 audit backlog.

- **matplotlib fully removed from the rendering path** — the hidden legacy figure that was still being built and rebuilt on every sample-rate/channel/display change is gone (~1,100 lines deleted from the window code). pyqtgraph/OpenGL is the sole renderer; matplotlib remains only as the source of the inferno colormap table. Amp-zoom persistence across Config↔View switches now reads from the live pyqtgraph viewbox (it had silently read a hidden, never-shown axis since the v3.0 port).
- **Busy cursor during device operations** — changing the input device, sample rate, channel mode, or WAV-simulation file shows a wait cursor for the duration of the multi-second rebuild instead of appearing frozen.
- **Auto-calibrate accuracy** — calibration now accumulates the per-chunk *trigger envelope* peak on the DSP thread (every chunk contributes, same statistic the trigger compares) instead of sampling the display buffer every 100 ms, which missed most chunks and biased thresholds low.
- **Collision-safe WAV publishing** — a filename collision at publish time gets a `_rNN` de-dup token instead of silently overwriting an earlier recording.
- **Findable error log in packaged builds** — `chirp_errors.log` is written next to the executable in frozen builds instead of whatever working directory the shortcut supplied.

## What's New in v3.1.0

A reliability + efficiency release following a full audit of the v3.0.0 data path, focused on "multi-stream recording just works": no sample loss, no hangs, bounded memory.

- **Streaming recording wired into the trigger** — events now append to their WAV incrementally as audio arrives (`StreamingWavWriter`), keeping only a bounded pending tail in RAM (≈ hold + post-trigger) instead of buffering up to `max_rec` seconds per event. Output is byte-identical to the previous buffered path (which remains as fallback), including force-split `partNN` files.
- **Capture-ring hardening** — the SPSC ring no longer lets the producer touch the consumer's cursor on overrun; reads are re-validated after copying, so a pathological overrun becomes a counted drop instead of silently corrupted audio.
- **Lock-free monitor loopback** — the audio-monitor ring buffer no longer takes a lock (or allocates) inside the PortAudio input callback, removing a priority-inversion path that could lose input samples while monitoring.
- **Shutdown correctness** — the writer pool is shut down only after all entities close (a late flush could previously resurrect a worker pool that kept the process alive forever), and shutdown-time flushes land in the correct `ref_date` day-subfolder.
- **Capture-time-anchored timestamps** — recording onset timestamps (and filenames) are derived from a wall-clock ↔ sample-counter anchor, so a backlogged DSP thread can no longer stamp recordings late.
- **Audio-priority rendering in Config mode** — the same adaptive frame-skip View mode already had, plus per-pixel peak decimation *before* the dB conversion and data-upload skipping when no new audio arrived; per-tick render cost no longer scales with `display window × sample rate`.
- **Error telemetry fixes** — OS-level input overflows now reach `chirp_errors.log`; a dead ingest thread is detected and reported distinctly instead of masquerading as generic drops.
- **Spectrogram frequency axis fixed** — Config- and View-mode spectrograms now label the y-axis with actual frequencies (Mel/Log/Linear aware) instead of raw display-row indices.

## What's New in v3.0.0

A major re-architecture of the entire data path and rendering engine, focused on **zero dropped data** during multi-stream recording and on running the whole pipeline as cheaply as possible on common multi-core + GPU hardware. Verified by an automated soak test driving **8 concurrent streams** at real time with zero ring overruns, zero OS overflows, zero ingest errors, and gapless WAVs.

- **Lossless, realtime-safe capture** — the PortAudio callback no longer allocates, logs, or does any disk I/O; it does a single lock-free memcpy into a preallocated per-stream ring buffer. The old 200-chunk queue that silently dropped audio is gone. Logging moved entirely off the realtime and DSP threads onto a background writer thread — removing the in-callback disk write that was the root cause of cascading overflows.
- **Vectorized trigger + GIL-releasing FFT** — the threshold state machine's per-sample Python loops were replaced with vectorized NumPy run-length logic (~3,400× realtime per stream, behavior-identical), and the spectrogram FFT now uses `scipy.fft` (pocketfft), which releases the GIL so per-stream DSP genuinely parallelizes across cores.
- **Streaming WAV writes** — recordings are written incrementally to disk via `soundfile`/libsndfile instead of buffering the whole event in RAM, and remain byte-identical and atomically published (`.tmp` → fsync → rename).
- **pyqtgraph/OpenGL visualization** — both Config and View modes were ported off matplotlib to pyqtgraph with an OpenGL-composited viewport: cheap per-frame image blits (vs matplotlib's full-figure re-raster), auto-downsampled curves, off-screen tile culling, and an **audio-priority adaptive frame rate** that drops render frequency under load so audio capture is never starved. (Colormap mapping is done on the CPU; OpenGL handles the scene compositing.)
- **View Mode "Fit to screen"** — size the monitoring grid tiles to fit the window for the current column count, in addition to the existing column and per-tile height controls.
- **Software-render fallback** — a `use_opengl` toggle is persisted in the settings file for machines with problematic GPU drivers.

## What's New in v2.2.1

A single-issue patch release fixing a v2.1.0 regression that prevented threshold-triggered recording from firing on narrowband signals (pure tones, bandpassed bioacoustic calls, whistles).

- **Envelope-based amplitude trigger** — `amp_mask` is now built from the analytic-signal envelope (`|hilbert(filt)|`) rather than `|filt|`. The pre-fix per-sample compare dipped to zero at every waveform zero crossing, which reset `_above_streak` in the state machine on every half-cycle; a sustained 1 kHz sine whose envelope was 5× threshold could never satisfy `min_cross` because the raw-amplitude compare only ever accumulated ~20 consecutive above samples between zero crossings. The yellow detect strip also flickered at the signal frequency instead of being solid during a tone. Regression introduced in v2.1.0 by the sample-accurate state-machine rewrite (#15); pre-v2.1 versions compared a chunk-level peak and were unaffected. Fix lives in the new `chirp/dsp/envelope.py` module; regression test in `tests/test_envelope_trigger.py`.

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
