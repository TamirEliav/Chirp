"""Generate the Chirp app icon (assets/chirp.png + assets/chirp.ico).

A rising chirp waveform (frequency sweeping up) in Catppuccin teal on
the app's dark base, with a soft mauve spectrogram-sweep glow behind it
and a peach trigger-threshold dash. Run from the repo root:

    python scripts/make_icon.py

Regenerating is deterministic — tweak and re-run freely.
"""

import math
import os

from PIL import Image, ImageDraw

# Catppuccin Mocha
BASE = (30, 30, 46, 255)        # #1e1e2e
SURFACE = (69, 71, 90, 255)     # #45475a
TEAL = (148, 226, 213, 255)     # #94e2d5
MAUVE = (203, 166, 247, 255)    # #cba6f7
PEACH = (250, 179, 135, 255)    # #fab387

S = 512  # master render size (downscaled for the .ico levels)


def draw_icon() -> Image.Image:
    img = Image.new('RGBA', (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    # Rounded-square background with a subtle border.
    r = S // 5
    d.rounded_rectangle([8, 8, S - 8, S - 8], radius=r, fill=BASE,
                        outline=SURFACE, width=6)

    # Soft mauve "spectrogram sweep": ascending translucent bars.
    bar_w = S // 14
    for i, (x_frac, h_frac) in enumerate(
            ((0.18, 0.22), (0.34, 0.34), (0.50, 0.48), (0.66, 0.62),
             (0.82, 0.78))):
        x = int(S * x_frac)
        h = int(S * 0.58 * h_frac)
        y1 = int(S * 0.80)
        col = MAUVE[:3] + (34 + i * 6,)
        d.rounded_rectangle([x - bar_w // 2, y1 - h, x + bar_w // 2, y1],
                            radius=bar_w // 2, fill=col)

    # Peach threshold dash across the middle.
    ty = int(S * 0.50)
    dash, gap, x = int(S * 0.055), int(S * 0.035), int(S * 0.14)
    while x < S * 0.86:
        d.line([x, ty, min(int(S * 0.86), x + dash), ty],
               fill=PEACH[:3] + (150,), width=8)
        x += dash + gap

    # The chirp: sine with quadratically increasing frequency, drawn as
    # a thick teal polyline with amplitude growing toward the right.
    pts = []
    x0, x1 = S * 0.12, S * 0.88
    for i in range(801):
        t = i / 800.0
        x = x0 + (x1 - x0) * t
        phase = 2.0 * math.pi * (1.0 * t + 2.1 * t * t)
        amp = S * (0.07 + 0.19 * t)
        y = S * 0.50 - amp * math.sin(phase)
        pts.append((x, y))
    d.line(pts, fill=TEAL, width=int(S * 0.045), joint='curve')

    return img


def main():
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'assets')
    os.makedirs(out_dir, exist_ok=True)
    img = draw_icon()
    png = os.path.join(out_dir, 'chirp.png')
    ico = os.path.join(out_dir, 'chirp.ico')
    img.resize((256, 256), Image.LANCZOS).save(png)
    img.save(ico, sizes=[(16, 16), (24, 24), (32, 32), (48, 48),
                         (64, 64), (128, 128), (256, 256)])
    print(f'wrote {png} and {ico}')


if __name__ == '__main__':
    main()
