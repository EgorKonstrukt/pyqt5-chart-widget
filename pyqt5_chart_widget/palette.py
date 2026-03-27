from __future__ import annotations
from typing import List
from PyQt5.QtGui import QColor

_PALETTE: List[str] = [
    "#e74c3c",
    "#3498db",
    "#2ecc71",
    "#f39c12",
    "#9b59b6",
    "#1abc9c",
    "#e67e22",
    "#34495e",
    "#e91e63",
    "#00bcd4",
    "#8bc34a",
    "#ff5722",
]

_line_idx = 0
_scatter_idx = 0


def next_line_color() -> str:
    global _line_idx
    c = _PALETTE[_line_idx % len(_PALETTE)]
    _line_idx += 1
    return c


def next_scatter_color() -> str:
    global _scatter_idx
    offset = len(_PALETTE) // 3
    c = _PALETTE[(_scatter_idx + offset) % len(_PALETTE)]
    _scatter_idx += 1
    return c


def reset_colors():
    global _line_idx, _scatter_idx
    _line_idx = 0
    _scatter_idx = 0


def set_palette(colors: List[str]):
    global _PALETTE
    _PALETTE = list(colors)
    reset_colors()


def contrast_color(base: QColor) -> QColor:
    lum = 0.299 * base.redF() + 0.587 * base.greenF() + 0.114 * base.blueF()
    return QColor("#000000") if lum > 0.5 else QColor("#ffffff")
