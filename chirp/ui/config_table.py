"""All-streams configuration table (4b / TODO#5, v3.3.0).

A modeless dialog showing every stream's parameters side by side —
rows = parameters (grouped), columns = streams. Most cells are editable:
an edit is validated by parameter kind and applied to the entity through
the same code paths the per-stream widgets use (the window supplies an
``apply_cb`` for that, so side effects like FFT rebuilds and dirty
marking stay in one place).

Structural parameters (device, sample rate, channel mode, input source)
are shown read-only in this first version — changing them requires a
capture rebuild that belongs to the per-stream Settings panel.

Cells are plain text for simplicity; choice-rows validate against their
allowed values (listed in the row tooltip) and boolean rows accept
true/false/1/0/yes/no. Invalid input reverts the cell.
"""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QDialog, QHBoxLayout, QPushButton,
                             QTableWidget, QTableWidgetItem, QVBoxLayout)

from chirp.constants import C
from chirp.recording.entity import RecordingEntity


# (key, label, kind, extra) — kind ∈ {'float', 'str', 'choice', 'bool',
# 'readonly'}; extra = allowed values for 'choice'. ``None`` key rows are
# group separators.
_ROWS: list = [
    (None, '— Structure —', None, None),
    ('device_name',   'Device',       'readonly', None),
    ('sample_rate',   'Sample Rate',  'readonly', None),
    ('channel_mode',  'Channels',     'readonly', None),
    ('input_source',  'Input Source', 'readonly', None),
    (None, '— Recording —', None, None),
    ('stream_enabled', 'Enabled', 'bool', None),
    ('rec_mode',      'Rec Mode',     'choice', ('Triggered', 'Continuous')),
    ('threshold',     'Threshold',    'float', (0.0, 1.0)),
    ('min_cross_sec', 'Min Cross (s)', 'float', (0.0, 60.0)),
    ('min_total_cross_sec', 'Min Total Cross (s)', 'float', (0.0, 3600.0)),
    ('hold_sec',      'Hold (s)',     'float', (0.0, 60.0)),
    ('pre_trig_sec',  'Pre-Trigger (s)', 'float', (0.0, 60.0)),
    ('post_trig_sec', 'Post-Trigger (s)', 'float', (0.0, 60.0)),
    ('max_rec_sec',   'Max Rec (s)',  'float', (1.0, 3600.0)),
    ('trigger_mode',  'Stereo Trigger', 'choice',
     ('Left Channel', 'Right Channel', 'Any Channel', 'Both Channels',
      'Average')),
    (None, '— Band Filter —', None, None),
    ('freq_filter_enabled', 'Band Filter', 'bool', None),
    ('freq_lo',       'Filter Lo (Hz)', 'float', (1.0, 500000.0)),
    ('freq_hi',       'Filter Hi (Hz)', 'float', (1.0, 500000.0)),
    (None, '— Spectral Trigger —', None, None),
    ('spectral_trigger_mode', 'Detect Mode', 'choice',
     ('Amplitude Only', 'Spectral Only', 'Amp AND Spectral',
      'Amp OR Spectral')),
    ('spectral_threshold', 'Entropy Thr', 'float', (0.0, 1.0)),
    ('entropy_min_cross_sec', 'Entropy Min Dur (s)', 'float', (0.0, 10.0)),
    (None, '— Display —', None, None),
    ('display_mode',  'Display Mode', 'choice',
     ('Spectrogram', 'Waveform', 'Both')),
    ('display_seconds', 'Window (s)', 'choice',
     tuple(str(s) for s in RecordingEntity.SUPPORTED_DISPLAY_SECONDS)),
    ('freq_scale',    'Freq Scale',   'choice', ('Linear', 'Log', 'Mel')),
    ('spec_nperseg',  'FFT Size',     'choice',
     ('256', '512', '1024', '2048', '4096')),
    ('spec_window',   'FFT Window',   'choice',
     ('hann', 'hamming', 'blackman', 'bartlett', 'flattop')),
    ('gain_db',       'Gain (dB)',    'float', (-60.0, 60.0)),
    ('db_floor',      'dB Floor',     'float', (-200.0, 0.0)),
    ('db_ceil',       'dB Ceil',      'float', (-100.0, 50.0)),
    ('display_freq_lo', 'Disp Freq Lo (Hz)', 'float', (0.0, 500000.0)),
    ('display_freq_hi', 'Disp Freq Hi (Hz)', 'float', (1.0, 500000.0)),
    ('amp_scale',     'Amp Scale',    'choice', ('linear', 'log')),
    (None, '— Output —', None, None),
    ('output_dir',    'Output Folder', 'str', None),
    ('filename_prefix', 'Prefix',     'str', None),
    ('filename_suffix', 'Suffix',     'str', None),
    ('dph_folder_prefix', 'Day-Folder Prefix', 'str', None),
]

_TRUE = {'1', 'true', 'yes', 'on'}
_FALSE = {'0', 'false', 'no', 'off'}


def _fmt(val) -> str:
    if isinstance(val, bool):
        return 'true' if val else 'false'
    if isinstance(val, float):
        return f'{val:g}'
    return '' if val is None else str(val)


class ConfigTableDialog(QDialog):
    """Modeless all-streams parameter table."""

    def __init__(self, window):
        super().__init__(window)
        self._win = window
        self.setWindowTitle('All Streams — Configuration Table')
        self.resize(760, 640)
        self.setModal(False)
        # QDialogs default to a title bar with only the useless '?'
        # (WhatsThis) button — nothing here sets WhatsThis text. Swap it
        # for proper minimize/maximize buttons; with many streams the
        # table is exactly the window you want maximized.
        self.setWindowFlags(
            (self.windowFlags()
             | Qt.WindowMinimizeButtonHint
             | Qt.WindowMaximizeButtonHint)
            & ~Qt.WindowContextHelpButtonHint)

        v = QVBoxLayout(self)
        self._table = QTableWidget()
        self._table.setStyleSheet(
            f'QTableWidget {{ background-color: {C["mantle"]}; '
            f'color: {C["text"]}; gridline-color: {C["surface0"]}; }}'
            f'QHeaderView::section {{ background-color: {C["surface0"]}; '
            f'color: {C["subtext"]}; padding: 3px; border: none; }}')
        v.addWidget(self._table)

        row = QHBoxLayout()
        btn_refresh = QPushButton('↻ Refresh')
        btn_refresh.setToolTip('Re-read all values from the streams')
        btn_refresh.clicked.connect(self.refresh)
        row.addWidget(btn_refresh)
        row.addStretch()
        btn_close = QPushButton('Close')
        btn_close.clicked.connect(self.close)
        row.addWidget(btn_close)
        v.addLayout(row)

        self._loading = False
        self._table.cellChanged.connect(self._on_cell_changed)

    # ── Populate ──────────────────────────────────────────────────────
    def showEvent(self, ev):
        self.refresh()
        super().showEvent(ev)

    def refresh(self):
        self._loading = True
        try:
            entities = self._win._entities
            t = self._table
            t.clear()
            t.setColumnCount(len(entities))
            t.setRowCount(len(_ROWS))
            t.setHorizontalHeaderLabels([e.name for e in entities])
            t.setVerticalHeaderLabels(
                [label for _key, label, _k, _x in _ROWS])
            for r, (key, label, kind, extra) in enumerate(_ROWS):
                for c, e in enumerate(entities):
                    if key is None:                    # group separator
                        item = QTableWidgetItem('')
                        item.setFlags(Qt.NoItemFlags)
                        item.setBackground(
                            self.palette().window().color().darker(110))
                        t.setItem(r, c, item)
                        continue
                    if key == 'device_name':
                        val = e.to_dict().get('device_name', '') \
                            if e.input_source == 'device' else e.wav_file_path
                    else:
                        val = getattr(e, key, '')
                    item = QTableWidgetItem(_fmt(val))
                    if kind == 'readonly':
                        item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
                        item.setForeground(Qt.gray)
                    if kind == 'choice' and extra:
                        item.setToolTip('One of: ' + ', '.join(extra))
                    elif kind == 'bool':
                        item.setToolTip('true / false')
                    t.setItem(r, c, item)
            t.resizeColumnsToContents()
        finally:
            self._loading = False

    # ── Edits ─────────────────────────────────────────────────────────
    def _on_cell_changed(self, row: int, col: int):
        if self._loading:
            return
        key, _label, kind, extra = _ROWS[row]
        if key is None or kind == 'readonly':
            return
        if not (0 <= col < len(self._win._entities)):
            return
        item = self._table.item(row, col)
        text = (item.text() or '').strip()
        value = self._parse(kind, extra, text)
        if value is None and kind != 'str':
            # Invalid — revert to the entity's current value.
            self._revert_cell(row, col, key)
            return
        if kind == 'str':
            value = text
        try:
            self._win._apply_table_edit(col, key, value)
        except Exception as exc:
            print(f'[ConfigTable] edit failed ({key}): {exc}')
        self._revert_cell(row, col, key)   # re-read the applied value

    @staticmethod
    def _parse(kind: str, extra, text: str):
        if kind == 'float':
            try:
                v = float(text)
            except ValueError:
                return None
            if extra:
                lo, hi = extra
                v = max(lo, min(hi, v))
            return v
        if kind == 'bool':
            low = text.lower()
            if low in _TRUE:
                return True
            if low in _FALSE:
                return False
            return None
        if kind == 'choice':
            if extra:
                # Case-insensitive match against the allowed values.
                for opt in extra:
                    if text.lower() == opt.lower():
                        return opt
            return None
        return text

    def _revert_cell(self, row: int, col: int, key: str):
        e = self._win._entities[col]
        self._loading = True
        try:
            self._table.item(row, col).setText(_fmt(getattr(e, key, '')))
        finally:
            self._loading = False
