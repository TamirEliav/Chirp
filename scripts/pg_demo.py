"""Phase 4 verification demo — launch the pyqtgraph/OpenGL renderer.

Runs N live streams (default 4) from a looping tone WAV through the real
capture→ring→DSP pipeline and renders each with the new pyqtgraph panel
at ~30 fps. Use this to confirm OpenGL renders correctly on your machine
before the full window integration.

    conda activate chirp
    python scripts/pg_demo.py            # 4 streams, synthesized tone
    python scripts/pg_demo.py 8          # 8 streams
    python scripts/pg_demo.py 4 my.wav   # 4 streams from your WAV

Close the window to stop. Watch the title bar — it reports the worst
ring-overrun / drop / ingest-error counts across all streams; they must
stay at 0.
"""

import math
import os
import sys
import tempfile

import numpy as np
import scipy.io.wavfile
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QGridLayout, QWidget

# Ensure the repo root is importable when run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chirp.constants import C
from chirp.recording.entity import RecordingEntity
from chirp.ui.pg_panel import StreamPlotPanel


def _synth_wav(path: str, sr: int = 44100, dur: float = 3.0) -> None:
    t = np.arange(int(sr * dur)) / sr
    # A chirp sweep so the spectrogram shows a moving feature.
    f0, f1 = 500.0, 8000.0
    sweep = np.sin(2 * np.pi * (f0 * t + (f1 - f0) / (2 * dur) * t * t))
    tone = (0.4 * sweep).astype(np.float32)
    scipy.io.wavfile.write(path, sr, (tone * 32767.0).astype(np.int16))


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    if len(sys.argv) > 2:
        wav_path = sys.argv[2]
    else:
        wav_path = os.path.join(tempfile.gettempdir(), 'chirp_pg_demo.wav')
        _synth_wav(wav_path)

    app = QApplication(sys.argv[:1])

    win = QWidget()
    win.setWindowTitle(f'Chirp pyqtgraph/OpenGL demo — {n} streams')
    win.setStyleSheet(f'background-color: {C["base"]};')
    grid = QGridLayout(win)
    grid.setContentsMargins(4, 4, 4, 4)
    grid.setSpacing(4)

    entities = []
    panels = []
    cols = max(1, int(math.ceil(math.sqrt(n))))
    for i in range(n):
        e = RecordingEntity(name=f'S{i}', device_id=None)
        ok, warn = e.use_wav_file(wav_path, loop=True)
        if not ok:
            print(f'stream {i}: could not open {wav_path}: {warn}')
            return 1
        e.start_acq()
        entities.append(e)
        panel = StreamPlotPanel(use_opengl=True, show_waveform=False)
        panels.append(panel)
        grid.addWidget(panel, i // cols, i % cols)

    def tick():
        worst = (0, 0, 0)
        for e, panel in zip(entities, panels):
            panel.update_from_entity(e)
            worst = (
                max(worst[0], e.ring.overrun_count_total),
                max(worst[1], e.capture.os_drop_count_total),
                max(worst[2], e.ingest_error_count_total),
            )
        win.setWindowTitle(
            f'Chirp pyqtgraph/OpenGL demo — {n} streams | '
            f'overruns={worst[0]} os_drops={worst[1]} ingest_err={worst[2]}')

    timer = QTimer()
    timer.setInterval(33)  # ~30 fps
    timer.timeout.connect(tick)
    timer.start()

    win.resize(1200, 800)
    win.show()
    try:
        return app.exec_()
    finally:
        timer.stop()
        for e in entities:
            try:
                e.stop_acq()
                e.close()
            except Exception:
                pass


if __name__ == '__main__':
    raise SystemExit(main())
