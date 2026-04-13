# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Chirp is a real-time sound analysis and threshold-triggered recording desktop app for bioacoustics research. Built with Python/PyQt5. Single-file application (`chirp.py`, ~4200 lines). Current version: v1.2.0.

## Running the App

```bash
conda activate chirp
python chirp.py
```

## Dependencies

```bash
pip install -r requirements.txt
```

Stack: sounddevice (audio I/O), numpy, scipy (DSP/filtering/WAV), matplotlib (spectrogram rendering), PyQt5 (GUI). Conda env name: `chirp`.

## Building (PyInstaller)

The app is distributed as `dist/Chirp.exe` via PyInstaller. Build artifacts go in `build/` and `dist/`.

## Tests

No test infrastructure exists.

## Architecture

The entire application lives in `chirp.py` as a monolith. Key classes in dependency order:

1. **AudioCapture** — Wraps `sounddevice.InputStream`; callback enqueues 1024-sample chunks to a queue.
2. **SpectrogramAccumulator** — Overlapped FFT computation (configurable window function and FFT size). Returns both dB column and linear magnitude (the latter used for spectral entropy).
3. **BandpassFilter** — 4th-order Butterworth IIR filter with lazy coefficient redesign.
4. **ThresholdRecorder** — State machine (IDLE → PENDING → RECORDING) managing threshold-triggered WAV recording with pre-trigger buffer, hold, and post-trigger. WAV writes happen on a daemon thread.
5. **RecordingEntity** — Central data model for one audio stream. Owns an AudioCapture, SpectrogramAccumulator, BandpassFilter, ThresholdRecorder, and ring buffers (amplitude, spectrogram, entropy). `ingest_chunk()` is the main processing method. Also computes spectral entropy and applies the spectral trigger logic. Serializable to/from dict for config persistence.
6. **MiniAmplitudeWidget** / **RecordingSidebarItem** / **RecordingSidebar** — Sidebar UI widgets for multi-stream management.
7. **ChirpWindow** (QMainWindow) — Top-level orchestrator. Manages multiple RecordingEntities, matplotlib-based visualization with blitting, and config file I/O. Supports three visualization modes (Spectrogram, Waveform, Both) and two layout modes (Config mode for editing, View mode for monitoring grid). Dynamic subplot layout builds axes based on display mode and whether spectral entropy is active.

### Data Flow

```
AudioCapture callback → queue → RecordingEntity.ingest_chunk() → {filter, FFT, spectral entropy, ring buffers, trigger state machine} → ChirpWindow._update_plot() (50ms timer) → matplotlib blit
```

### Key Design Details

- **Ring buffers** use modulo-arithmetic write heads (`write_head`, `col_head`) for O(1) circular writes. Includes amplitude, spectrogram, and entropy buffers.
- **Frequency scales**: Linear, Log, and Mel modes via lookup-table interpolation in `rebuild_freq_mapping()`.
- **Spectral entropy trigger**: Normalized Shannon entropy (0=pure tone, 1=white noise) computed from FFT magnitudes. Four modes: Amplitude Only, Spectral Only, Amp AND Spectral, Amp OR Spectral. Triggers when entropy falls *below* threshold. Entropy trace plot with draggable threshold appears when a spectral mode is active.
- **Stereo trigger modes**: Left Channel, Right Channel, Any Channel, Both Channels, Average — applies to both amplitude and entropy triggers.
- **Auto-calibrate**: Measures ambient noise to set threshold automatically.
- **Display modes**: `display_mode` on RecordingEntity controls subplot layout — 'Spectrogram' (default), 'Waveform' (raw signed audio in teal), or 'Both'. The subplot grid is rebuilt dynamically in `_rebuild_axes()`.
- **Config format**: JSON with a `recordings` array and `view_mode` object. Also reads legacy `.chirp` files. v1.1.0 added `spectral_trigger_mode`, `spectral_threshold`, and `display_mode` to serialization.
- **Styling**: Catppuccin Mocha dark theme via embedded QSS; Consolas monospace font for labels. Teal (`#94e2d5`) for waveform, peach (`#fab387`) for entropy threshold.
- **Sample rate changes** require full pipeline rebuild (capture, filters, buffers, display).
- **Saturation detection**: clips at ≥ 0.99 of full scale, turns amplitude plot red.
- **Scroll-wheel zoom**: Mouse-centered zoom on entropy, waveform, and spectrogram axes.
- **User manual**: `manual.html` is a standalone HTML manual included in the repo.
