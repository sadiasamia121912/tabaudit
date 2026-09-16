"""Render real rich output to a PNG that looks like the terminal it came from.

Not a mock-up: the renderables are produced by tabaudit itself, and every colour is the
style rich would have printed, resolved to RGB. Pillow just draws the segments.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from rich.console import Console
from rich.segment import Segment

FONT = "C:/Windows/Fonts/consola.ttf"
SYMBOL = "C:/Windows/Fonts/seguisym.ttf"  # Consolas has no U+2714 etc.
FALLBACK = set("✔✓✖✗")
FONT_BOLD = "C:/Windows/Fonts/consolab.ttf"
SIZE = 22
PAD = 34
CHROME = 44  # title-bar height
BG = (13, 17, 23)
BAR = (22, 27, 34)
FG = (201, 209, 217)
DOTS = ((255, 95, 86), (255, 189, 46), (39, 201, 63))


def render(renderables, out_path: Path, width: int = 92, title: str = "") -> Path:
    console = Console(width=width, color_system="truecolor", record=False, file=None)
    lines: list[list[Segment]] = []
    for r in renderables:
        segs = console.render(r, console.options.update(width=width))
        lines.extend(Segment.split_and_crop_lines(segs, width, include_new_lines=False))

    font = ImageFont.truetype(FONT, SIZE)
    bold = ImageFont.truetype(FONT_BOLD, SIZE)
    symbol = ImageFont.truetype(SYMBOL, SIZE - 3)
    char_w = font.getlength("M")
    line_h = int(SIZE * 1.45)

    img_w = int(char_w * width) + PAD * 2
    img_h = line_h * len(lines) + PAD * 2 + CHROME
    img = Image.new("RGB", (img_w, img_h), BG)
    d = ImageDraw.Draw(img)

    # window chrome
    d.rectangle([0, 0, img_w, CHROME], fill=BAR)
    for i, colour in enumerate(DOTS):
        cx = 22 + i * 22
        d.ellipse([cx - 6, CHROME // 2 - 6, cx + 6, CHROME // 2 + 6], fill=colour)
    if title:
        d.text((cx + 26, CHROME // 2 - SIZE // 2 - 1), title, font=font, fill=(110, 118, 129))

    y = CHROME + PAD
    for line in lines:
        x = PAD
        for seg in line:
            if not seg.text:
                continue
            style = seg.style
            colour = FG
            if style is not None and style.color is not None:
                t = style.color.get_truecolor(foreground=True)
                colour = (t.red, t.green, t.blue)
            if style is not None and style.bgcolor is not None:
                t = style.bgcolor.get_truecolor(foreground=False)
                w = char_w * len(seg.text)
                d.rectangle([x, y - 2, x + w, y + line_h - 2], fill=(t.red, t.green, t.blue))
            f = bold if (style is not None and style.bold) else font
            for ch in seg.text:
                if ch in FALLBACK:
                    # centre the fallback glyph in one cell so the columns stay aligned
                    gw = symbol.getlength(ch)
                    d.text((x + (char_w - gw) / 2, y - 2), ch, font=symbol, fill=colour)
                else:
                    d.text((x, y), ch, font=f, fill=colour)
                x += char_w
        y += line_h

    img.save(out_path)
    return out_path
