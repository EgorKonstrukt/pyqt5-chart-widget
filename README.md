# pyqt5-chart-widget

A self-contained, dependency-free chart widget for PyQt5. No matplotlib, no pyqtgraph — just one file you drop into your project.

Built for an industrial powder feeder control application where the charting library needed to be small, fast, and not require a science degree to integrate.

---

## What it does

- Line plots and scatter series on the same canvas
- Mouse pan (drag) and scroll-to-zoom, centered on cursor
- Crosshair with snap-to-nearest-point and tangent line
- Inline analytics panel (min/max/mean/std per series)
- Infinite reference lines (horizontal or vertical)
- Export to CSV or PNG
- Adapts to system light/dark theme via QPalette
- i18n-ready — pass any `tr()` function or fall back to built-in English strings

---

## Install

```bash
pip install pyqt5-chart-widget
```

Requires Python ≥ 3.8 and PyQt5 ≥ 5.15.

---

## Quickstart

```python
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from pyqt5_chart_widget import ChartWidget

app = QApplication(sys.argv)
win = QMainWindow()

chart = ChartWidget()
chart.setLabel("left",   "Flow, g/min")
chart.setLabel("bottom", "RPM")

line    = chart.plot(color="#e74c3c", width=2)
scatter = chart.addScatter(size=10, color="#27ae60")

xs = [i * 100 for i in range(20)]
ys = [x * 0.05 + x ** 1.1 * 0.0003 for x in xs]
line.setData(xs=xs, ys=ys)
scatter.setData(x=xs[::3], y=ys[::3])

win.setCentralWidget(chart)
win.resize(800, 500)
win.show()
sys.exit(app.exec_())
```

---

## API

### ChartWidget

| Method | Description |
|--------|-------------|
| `plot(color, width)` | Add a line series, returns `LineItem` |
| `addScatter(size, color)` | Add a scatter series, returns `ScatterItem` |
| `addLine(y=, x=, color, width, dashed)` | Add a horizontal or vertical reference line |
| `setLabel(side, text)` | Set axis label — `side` is `"left"` or `"bottom"` |
| `autofit()` | Fit view to data bounds |
| `exportCsv()` | Opens save dialog, writes all series to CSV |
| `exportImage()` | Opens save dialog, saves canvas as PNG |

### LineItem

```python
item = chart.plot()
item.setData(xs=[...], ys=[...])
```

### ScatterItem

```python
item = chart.addScatter()
item.setData(x=[...], y=[...])
```

### InfLine

```python
hline = chart.addLine(y=100.0, color="#f39c12", dashed=True)
hline.setVisible(True)
hline.setValue(150.0)
```

---

## Interaction

| Action | Result |
|--------|--------|
| Scroll wheel | Zoom in/out centered on cursor |
| Left drag | Pan |
| Double-click | Reset view to data bounds |
| Hover | Crosshair + snap to nearest point + tangent |

---

## Internationalization

The widget ships with English strings but will use your app's `tr()` if you pass it in:

```python
# if your app already has an i18n module at import time, nothing to do —
# the widget tries `from i18n import tr` and falls back gracefully
```

String keys follow the pattern `chart_widget.*`, so add them to your locale files as needed. All keys and their English defaults are listed in the source near the top of `chart_widget.py`.

---

## Why not matplotlib / pyqtgraph?

Matplotlib embedded in Qt works, but it's slow to import, awkward to theme, and overkill when you just need one calibration curve on screen. Pyqtgraph is excellent but pulls in numpy as a hard dependency and has a learning curve.

This widget is ~600 lines, has zero runtime dependencies beyond PyQt5, and does exactly what it says on the tin.

---

## License

MIT