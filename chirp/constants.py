"""Chirp — module-level constants, color palette, and Qt stylesheet.

Everything here is pure data: audio/display defaults, recording defaults,
the Catppuccin Mocha palette, and the QSS theme string. Moved out of
the monolith during the Phase 1 refactor so that downstream modules
(audio, dsp, recording, ui) can import these without pulling in Qt or
matplotlib.

Note: this file intentionally has no imports other than what's needed
to build the constants themselves. Keep it that way — constants.py is
meant to be safe to import from every other module without circular
dependency risk.
"""

# ── Audio ──────────────────────────────────────────────────────────────────────
SAMPLE_RATE         = 44100
CHANNELS            = 1
CHUNK_FRAMES        = 1024
DTYPE               = 'float32'

# ── Display ────────────────────────────────────────────────────────────────────
DISPLAY_SECONDS     = 10.0
SPECTROGRAM_NPERSEG = 4096
COLORMAP            = 'inferno'
ANIMATION_INTERVAL  = 50
SPEC_DB_MIN         = -100.0
SPEC_DB_MAX         = 0.0
N_DISPLAY_ROWS      = 256
# Amplitude axis — log (dB) display. The envelope buffer is in linear
# [0, 1] full-scale units; for the dB view we plot 20*log10(|x|), with
# values below AMP_DB_MIN clamped at the floor so a momentary zero
# doesn't blow the line off the canvas.
AMP_DB_MIN          = -80.0
AMP_DB_MAX          =   0.0
AMP_DB_EPS          = 1e-4   # 10**(AMP_DB_MIN/20) — linear floor

# ── Recording defaults ─────────────────────────────────────────────────────────
DEFAULT_THRESHOLD   = 0.05
DEFAULT_MIN_CROSS   = 0.020
DEFAULT_HOLD        = 1.00
DEFAULT_PRE_TRIG    = 0.50
DEFAULT_POST_TRIG   = 0.50
DEFAULT_MAX_REC     = 60.0
RECORDINGS_DIR      = './recordings'
DEFAULT_FREQ_LO     = 1000.0
DEFAULT_FREQ_HI     = 8000.0

# ── Catppuccin Mocha palette ───────────────────────────────────────────────────
C = {
    'base':     '#1e1e2e',
    'mantle':   '#181825',
    'surface0': '#313244',
    'surface1': '#45475a',
    'surface2': '#585b70',
    'text':     '#cdd6f4',
    'subtext':  '#a6adc8',
    'blue':     '#89b4fa',
    'green':    '#a6e3a1',
    'red':      '#f38ba8',
    'yellow':   '#f9e2af',
    'mauve':    '#cba6f7',
    'pink':     '#f5c2e7',
    'teal':     '#94e2d5',
    'peach':    '#fab387',
}

QSS = f"""
QMainWindow, QWidget {{
    background-color: {C['base']};
    color: {C['text']};
    font-family: 'Segoe UI';
    font-size: 10pt;
}}
QGroupBox {{
    border: 1px solid {C['surface1']};
    border-radius: 6px;
    margin-top: 10px;
    padding: 8px 10px 6px 10px;
    font-weight: bold;
    color: {C['blue']};
    font-size: 8pt;
    letter-spacing: 1px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 4px;
    background-color: {C['base']};
}}
QPushButton {{
    background-color: {C['surface0']};
    border: 1px solid {C['surface1']};
    border-radius: 5px;
    padding: 9px 20px;
    color: {C['text']};
    min-width: 120px;
    min-height: 36px;
    font-size: 10pt;
}}
QPushButton:hover   {{ background-color: {C['surface1']}; }}
QPushButton:pressed {{ background-color: {C['surface2']}; }}
QPushButton#btn_start_acq               {{ border-color: {C['blue']};  color: {C['blue']};  }}
QPushButton#btn_start_acq[active=true]  {{ background-color: {C['blue']};  color: {C['mantle']}; font-weight: bold; }}
QPushButton#btn_start_rec               {{ border-color: {C['green']}; color: {C['green']}; }}
QPushButton#btn_start_rec[active=true]  {{ background-color: {C['green']}; color: {C['mantle']}; font-weight: bold; }}
QPushButton#btn_stop_acq  {{ border-color: {C['red']}; color: {C['red']}; }}
QPushButton#btn_stop_rec  {{ border-color: {C['red']}; color: {C['red']}; }}
QPushButton#btn_browse    {{ min-width: 80px; }}
QPushButton#btn_small     {{ min-width: 28px; min-height: 24px; padding: 2px; font-size: 9pt; }}
QSlider::groove:horizontal {{
    height: 4px; background: {C['surface1']}; border-radius: 2px;
}}
QSlider::sub-page:horizontal {{
    background: {C['mauve']}; border-radius: 2px;
}}
QSlider::handle:horizontal {{
    width: 16px; height: 16px; margin: -6px 0;
    background: {C['mauve']}; border-radius: 8px;
    border: 2px solid {C['base']};
}}
QSlider::handle:horizontal:hover {{ background: {C['pink']}; }}
QLineEdit, QDoubleSpinBox, QComboBox {{
    background-color: {C['mantle']};
    border: 1px solid {C['surface1']};
    border-radius: 4px;
    padding: 5px 8px;
    color: {C['text']};
    selection-background-color: {C['blue']};
    min-height: 28px;
}}
QLineEdit:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border-color: {C['blue']};
}}
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
    background-color: {C['surface0']};
    border: none;
    width: 18px;
}}
QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {{
    background-color: {C['surface1']};
}}
QComboBox::drop-down {{
    border: none;
    width: 24px;
    background-color: {C['surface0']};
}}
QComboBox QAbstractItemView {{
    background-color: {C['mantle']};
    border: 1px solid {C['surface1']};
    color: {C['text']};
    selection-background-color: {C['surface1']};
}}
QFrame[frameShape="4"] {{ color: {C['surface0']}; max-height: 1px; }}
QCheckBox {{ spacing: 6px; color: {C['text']}; }}
QCheckBox::indicator {{ width: 16px; height: 16px; border: 1px solid {C['surface1']}; border-radius: 3px; background: {C['mantle']}; }}
QCheckBox::indicator:checked {{ background: {C['mauve']}; border-color: {C['mauve']}; }}
QCheckBox::indicator:hover {{ border-color: {C['blue']}; }}
QLabel#param_label {{ color: {C['subtext']}; font-size: 9pt; }}
QLabel#status_on   {{ color: {C['green']}; font-weight: bold; font-size: 10pt; }}
QLabel#status_off  {{ color: {C['surface2']};              font-size: 10pt; }}
QLabel#trig_active {{ color: {C['red']};   font-weight: bold; font-size: 10pt; }}
QLabel#trig_idle   {{ color: {C['surface2']};              font-size: 10pt; }}
QScrollArea {{ border: none; }}
"""


__all__ = [
    # Audio
    "SAMPLE_RATE", "CHANNELS", "CHUNK_FRAMES", "DTYPE",
    # Display
    "DISPLAY_SECONDS", "SPECTROGRAM_NPERSEG", "COLORMAP",
    "ANIMATION_INTERVAL", "SPEC_DB_MIN", "SPEC_DB_MAX", "N_DISPLAY_ROWS",
    "AMP_DB_MIN", "AMP_DB_MAX", "AMP_DB_EPS",
    # Recording defaults
    "DEFAULT_THRESHOLD", "DEFAULT_MIN_CROSS", "DEFAULT_HOLD",
    "DEFAULT_POST_TRIG", "DEFAULT_MAX_REC", "DEFAULT_PRE_TRIG",
    "RECORDINGS_DIR", "DEFAULT_FREQ_LO", "DEFAULT_FREQ_HI",
    # Theme
    "C", "QSS",
]
