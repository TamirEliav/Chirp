"""Sidebar widgets — per-stream list on the left-hand side.

Extracted from the monolith in the Phase 1 refactor (plan: c08).
Contains MiniAmplitudeWidget (QPainter mini waveform),
RecordingSidebarItem (single row), and RecordingSidebar (the list).

c15 (#13) added a drop-indicator badge that flashes orange when the
capture callback drops chunks on queue.Full.
"""

import numpy as np

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QScrollArea, QFrame, QSizePolicy,
)
from PyQt5.QtCore import Qt, pyqtSignal, QPointF
from PyQt5.QtGui import QPainter, QColor, QPainterPath, QPen, QPolygonF

from chirp.constants import C

# ──────────────────────────────────────────────────────────────────────────────
# MiniAmplitudeWidget  — QPainter waveform preview for sidebar
# ──────────────────────────────────────────────────────────────────────────────
class MiniAmplitudeWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._data = np.zeros(200, dtype=np.float32)
        self._color = QColor(C['blue'])
        self.setFixedHeight(30)

    def set_data(self, data: np.ndarray):
        self._data = data
        self.update()

    def set_color(self, color: QColor):
        self._color = color

    def paintEvent(self, event):
        p = QPainter(self)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor(C['mantle']))

        data = self._data
        n = len(data)
        if n < 2 or w < 2:
            p.end()
            return

        y_max = max(float(data.max()), 0.01)
        # Pre-compute all points as QPolygonF (much faster than per-point path)
        xs = np.linspace(0, w, n, dtype=np.float64)
        ys = h - (data / y_max) * h
        pts = [QPointF(0.0, float(h))]
        pts.extend(QPointF(float(xs[i]), float(ys[i])) for i in range(n))
        pts.append(QPointF(float(w), float(h)))
        poly = QPolygonF(pts)

        color = QColor(self._color)
        color.setAlpha(80)
        path = QPainterPath()
        path.addPolygon(poly)
        path.closeSubpath()
        p.fillPath(path, color)
        pen = QPen(self._color)
        pen.setWidthF(1.0)
        p.setPen(pen)
        p.drawPolyline(poly)
        p.end()


# ──────────────────────────────────────────────────────────────────────────────
# RecordingSidebarItem
# ──────────────────────────────────────────────────────────────────────────────
class RecordingSidebarItem(QWidget):
    clicked  = pyqtSignal(int)
    move_up  = pyqtSignal(int)
    move_down = pyqtSignal(int)
    delete   = pyqtSignal(int)
    renamed  = pyqtSignal(int, str)

    def __init__(self, index: int, name: str, parent=None):
        super().__init__(parent)
        self._index = index
        self._selected = False
        self._editing = False
        self._last_status = None  # cached (acq, rec, trig) to avoid redundant setStyleSheet

        self.setFixedHeight(82)
        self.setCursor(Qt.PointingHandCursor)

        vbox = QVBoxLayout(self)
        vbox.setContentsMargins(6, 4, 6, 4)
        vbox.setSpacing(2)

        # Row 1: name + status dots
        row1 = QHBoxLayout()
        row1.setSpacing(6)
        self._name_label = QLabel(name)
        self._name_label.setStyleSheet(f'color: {C["text"]}; font-weight: bold; font-size: 9pt; border: none;')
        self._name_label.setToolTip('Double-click to rename this recording')
        self._name_edit = QLineEdit(name)
        self._name_edit.setStyleSheet(f'background: {C["surface0"]}; font-size: 9pt; min-height: 20px; padding: 1px 4px;')
        self._name_edit.hide()
        self._name_edit.setToolTip('Press Enter to confirm the new name')
        self._name_edit.editingFinished.connect(self._finish_edit)

        self._lbl_acq  = QLabel('\u25cf')
        self._lbl_rec  = QLabel('\u25cf')
        self._lbl_trig = QLabel('\u25cf')
        # #13 / c15: flashes orange when AudioCapture has dropped chunks
        # since the last UI tick. The badge is latched for a few ticks
        # after the last drop so brief stalls remain visible.
        self._lbl_drop = QLabel('!')
        self._drop_ttl = 0
        self._lbl_acq .setToolTip('Acquisition status (live monitoring)')
        self._lbl_rec .setToolTip('Recording status (threshold-triggered WAV saving enabled)')
        self._lbl_trig.setToolTip('Trigger status (currently writing to a WAV file)')
        self._lbl_drop.setToolTip(
            'Audio drops detected — the queue is overflowing because '
            'the UI/processing loop cannot keep up with the capture '
            'callback. Reduce the number of streams or lower the '
            'sample rate.')
        for lbl in (self._lbl_acq, self._lbl_rec, self._lbl_trig):
            lbl.setFixedWidth(12)
            lbl.setStyleSheet(f'color: {C["surface2"]}; font-size: 8pt;')
        self._lbl_drop.setFixedWidth(12)
        self._lbl_drop.setStyleSheet(f'color: {C["surface2"]}; font-weight: bold; font-size: 9pt;')

        row1.addWidget(self._name_label)
        row1.addWidget(self._name_edit)
        row1.addStretch()
        row1.addWidget(self._lbl_drop)
        row1.addWidget(self._lbl_acq)
        row1.addWidget(self._lbl_rec)
        row1.addWidget(self._lbl_trig)
        vbox.addLayout(row1)

        # Row 2: mini amplitude
        self._mini_amp = MiniAmplitudeWidget()
        vbox.addWidget(self._mini_amp)

        # Row 3: move/delete buttons
        row3 = QHBoxLayout()
        row3.setSpacing(4)
        btn_up = QPushButton('\u25b2')
        btn_dn = QPushButton('\u25bc')
        btn_del = QPushButton('\u2715')
        for b in (btn_up, btn_dn, btn_del):
            b.setObjectName('btn_small')
            b.setFixedSize(24, 20)
        btn_up .setToolTip('Move this recording up in the list')
        btn_dn .setToolTip('Move this recording down in the list')
        btn_del.setToolTip('Delete this recording')
        btn_up.clicked.connect(lambda: self.move_up.emit(self._index))
        btn_dn.clicked.connect(lambda: self.move_down.emit(self._index))
        btn_del.clicked.connect(lambda: self.delete.emit(self._index))
        row3.addWidget(btn_up)
        row3.addWidget(btn_dn)
        row3.addWidget(btn_del)
        row3.addStretch()
        vbox.addLayout(row3)

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, v):
        self._index = v

    def set_selected(self, sel: bool):
        self._selected = sel
        if sel:
            self.setStyleSheet(
                f'background: {C["surface0"]}; '
                f'border: 2px solid {C["blue"]}; border-radius: 6px; '
                f'border-left: 4px solid {C["blue"]};')
            self._name_label.setStyleSheet(
                f'color: {C["blue"]}; font-weight: bold; font-size: 9pt; border: none;')
        else:
            self.setStyleSheet(
                f'background: {C["mantle"]}; '
                f'border: 1px solid {C["surface1"]}; border-radius: 6px;')
            self._name_label.setStyleSheet(
                f'color: {C["text"]}; font-weight: bold; font-size: 9pt; border: none;')

    def set_name(self, name: str):
        self._name_label.setText(name)

    def update_status(self, acq: bool, rec: bool, trig: bool):
        key = (acq, rec, trig)
        if key == self._last_status:
            return
        self._last_status = key
        self._lbl_acq .setStyleSheet(f'color: {C["blue"] if acq else C["surface2"]}; font-size: 8pt;')
        self._lbl_rec .setStyleSheet(f'color: {C["green"] if rec else C["surface2"]}; font-size: 8pt;')
        self._lbl_trig.setStyleSheet(f'color: {C["red"] if trig else C["surface2"]}; font-size: 8pt;')

    def update_mini_amp(self, data: np.ndarray):
        self._mini_amp.set_data(data)

    def notify_drops(self, n_drops: int):
        """Latch the drop indicator. Called once per UI tick with the
        number of dropped chunks since the last tick. The badge stays
        lit for `_DROP_LATCH_TICKS` ticks after the most recent drop
        so brief stalls remain visible.
        """
        if n_drops > 0:
            self._drop_ttl = 20  # ~1s at the default 50ms tick
        elif self._drop_ttl > 0:
            self._drop_ttl -= 1
        active = self._drop_ttl > 0
        color = C['peach'] if active else C['surface2']
        self._lbl_drop.setStyleSheet(
            f'color: {color}; font-weight: bold; font-size: 9pt;')

    def mouseDoubleClickEvent(self, event):
        self._start_edit()

    def mousePressEvent(self, event):
        if not self._editing:
            self.clicked.emit(self._index)

    def _start_edit(self):
        self._editing = True
        self._name_edit.setText(self._name_label.text())
        self._name_label.hide()
        self._name_edit.show()
        self._name_edit.setFocus()
        self._name_edit.selectAll()

    def _finish_edit(self):
        if not self._editing:
            return
        self._editing = False
        new_name = self._name_edit.text().strip()
        if not new_name:
            new_name = self._name_label.text()
        self._name_label.setText(new_name)
        self._name_edit.hide()
        self._name_label.show()
        self.renamed.emit(self._index, new_name)


# ──────────────────────────────────────────────────────────────────────────────
# RecordingSidebar
# ──────────────────────────────────────────────────────────────────────────────
class RecordingSidebar(QWidget):
    selection_changed = pyqtSignal(int)
    add_requested     = pyqtSignal()
    delete_requested  = pyqtSignal(int)
    move_requested    = pyqtSignal(int, int)   # (index, direction: -1 up, +1 down)
    item_renamed      = pyqtSignal(int, str)
    start_all_acq     = pyqtSignal()
    stop_all_acq      = pyqtSignal()
    start_all_rec     = pyqtSignal()
    stop_all_rec      = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(250)
        self._items: list[RecordingSidebarItem] = []

        vbox = QVBoxLayout(self)
        vbox.setContentsMargins(4, 4, 4, 4)
        vbox.setSpacing(4)

        # All-recordings transport
        all_row1 = QHBoxLayout()
        all_row1.setSpacing(4)
        btn_sa = QPushButton('Start All Acq')
        btn_xa = QPushButton('Stop All Acq')
        btn_sa.setObjectName('btn_start_acq')
        btn_xa.setObjectName('btn_stop_acq')
        for b in (btn_sa, btn_xa):
            b.setFixedHeight(28)
            b.setStyleSheet(b.styleSheet() + 'min-width: 0px; padding: 4px 8px; font-size: 9pt;')
        btn_sa.setToolTip('Start audio acquisition (live monitoring) for all recordings')
        btn_xa.setToolTip('Stop audio acquisition for all recordings')
        btn_sa.clicked.connect(self.start_all_acq.emit)
        btn_xa.clicked.connect(self.stop_all_acq.emit)
        all_row1.addWidget(btn_sa)
        all_row1.addWidget(btn_xa)
        vbox.addLayout(all_row1)

        all_row2 = QHBoxLayout()
        all_row2.setSpacing(4)
        btn_sr = QPushButton('Start All Rec')
        btn_xr = QPushButton('Stop All Rec')
        btn_sr.setObjectName('btn_start_rec')
        btn_xr.setObjectName('btn_stop_rec')
        for b in (btn_sr, btn_xr):
            b.setFixedHeight(28)
            b.setStyleSheet(b.styleSheet() + 'min-width: 0px; padding: 4px 8px; font-size: 9pt;')
        btn_sr.setToolTip('Enable threshold-triggered WAV recording for all recordings')
        btn_xr.setToolTip('Disable threshold-triggered WAV recording for all recordings')
        btn_sr.clicked.connect(self.start_all_rec.emit)
        btn_xr.clicked.connect(self.stop_all_rec.emit)
        all_row2.addWidget(btn_sr)
        all_row2.addWidget(btn_xr)
        vbox.addLayout(all_row2)

        btn_add = QPushButton('+  Add Recording')
        btn_add.setObjectName('btn_browse')
        btn_add.setFixedHeight(32)
        btn_add.setToolTip('Add a new recording stream')
        btn_add.clicked.connect(self.add_requested.emit)
        vbox.addWidget(btn_add)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scroll_widget = QWidget()
        self._scroll_layout = QVBoxLayout(self._scroll_widget)
        self._scroll_layout.setContentsMargins(0, 0, 0, 0)
        self._scroll_layout.setSpacing(4)
        self._scroll_layout.addStretch()
        scroll.setWidget(self._scroll_widget)
        vbox.addWidget(scroll)

    def add_item(self, name: str) -> int:
        idx = len(self._items)
        item = RecordingSidebarItem(idx, name)
        item.clicked.connect(self._on_item_clicked)
        item.move_up.connect(lambda i: self.move_requested.emit(i, -1))
        item.move_down.connect(lambda i: self.move_requested.emit(i, 1))
        item.delete.connect(self.delete_requested.emit)
        item.renamed.connect(self.item_renamed.emit)
        self._scroll_layout.insertWidget(idx, item)
        self._items.append(item)
        return idx

    def remove_item(self, idx: int):
        if 0 <= idx < len(self._items):
            item = self._items.pop(idx)
            self._scroll_layout.removeWidget(item)
            item.deleteLater()
            self._reindex()

    def select(self, idx: int):
        for i, item in enumerate(self._items):
            item.set_selected(i == idx)

    def swap_items(self, idx_a: int, idx_b: int):
        if not (0 <= idx_a < len(self._items) and 0 <= idx_b < len(self._items)):
            return
        # Swap in list
        self._items[idx_a], self._items[idx_b] = self._items[idx_b], self._items[idx_a]
        # Remove both from layout (higher index first to preserve lower)
        hi, lo = max(idx_a, idx_b), min(idx_a, idx_b)
        self._scroll_layout.removeWidget(self._items[hi])
        self._scroll_layout.removeWidget(self._items[lo])
        # Re-insert at correct positions
        self._scroll_layout.insertWidget(lo, self._items[lo])
        self._scroll_layout.insertWidget(hi, self._items[hi])
        self._reindex()

    def update_item_status(self, idx: int, acq: bool, rec: bool, trig: bool):
        if 0 <= idx < len(self._items):
            self._items[idx].update_status(acq, rec, trig)

    def update_item_amp(self, idx: int, data: np.ndarray):
        if 0 <= idx < len(self._items):
            self._items[idx].update_mini_amp(data)

    def update_item_drops(self, idx: int, n_drops: int):
        if 0 <= idx < len(self._items):
            self._items[idx].notify_drops(n_drops)

    def update_item_name(self, idx: int, name: str):
        if 0 <= idx < len(self._items):
            self._items[idx].set_name(name)

    def clear_all(self):
        for item in self._items:
            self._scroll_layout.removeWidget(item)
            item.deleteLater()
        self._items.clear()

    def count(self):
        return len(self._items)

    def _reindex(self):
        for i, item in enumerate(self._items):
            item.index = i

    def _on_item_clicked(self, idx: int):
        self.selection_changed.emit(idx)
