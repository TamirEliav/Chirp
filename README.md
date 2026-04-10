# Chirp

**Real-time sound analysis and recording tool for researchers.**

Chirp is a desktop application for multi-stream audio monitoring, visualization, and threshold-triggered recording. It was designed with bioacoustics research in mind but works for any audio analysis task.

![Version](https://img.shields.io/badge/Version-v1.1.0-orange) ![Python](https://img.shields.io/badge/Python-3.11+-blue) ![PyQt5](https://img.shields.io/badge/GUI-PyQt5-green) ![License](https://img.shields.io/badge/License-MIT-yellow)

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
