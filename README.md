# pyqt5-chart-widget

A self-contained, dependency-free chart widget for PyQt5. No matplotlib, no pyqtgraph — drop the package into your project and go.

Built for an industrial powder feeder control application where the charting library needed to be small, fast, and not require a science degree to integrate.

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

xs = [i * 100 for i in range(20)]
ys = [x * 0.05 + x ** 1.1 * 0.0003 for x in xs]

line = chart.plot(label="Pump A")
line.setData(xs=xs, ys=ys)

scatter = chart.addScatter(size=10, label="Calibration")
scatter.setData(x=xs[::3], y=ys[::3])

fit = chart.addFit(line, mode_key="poly3", label="Polynomial fit")

win.setCentralWidget(chart)
win.resize(900, 500)
win.show()
sys.exit(app.exec_())
```

---

## API

### ChartWidget

```python
ChartWidget(
    parent=None,
    show_toolbar: bool = True,
    show_legend:  bool = False,
    show_sidebar: bool = False,
    font: QFont | None = None,
)
```

| Method | Description |
|--------|-------------|
| `plot(color, width, label, dashed)` | Add a line series → `_LineItem` |
| `addScatter(size, color, label)` | Add a scatter series → `_ScatterItem` |
| `addFit(source, mode_key, color, width, dashed, label)` | Add a fit curve for a series → `_FitItem` |
| `addLine(y=, x=, color, width, dashed)` | Add an infinite reference line → `_InfLine` |
| `setLabel(side, text)` | Set axis label — `"left"` or `"bottom"` |
| `autofit()` | Fit view to data bounds |
| `setToolbarVisible(visible)` | Show / hide the toolbar |
| `setSidebarVisible(visible)` | Show / hide the sidebar |
| `setLegendVisible(visible)` | Show / hide the legend overlay |
| `sidebar()` | Return the `SidebarLabel` instance (or `None`) |
| `removeItem(item)` | Remove any series or line |
| `clearAll()` | Remove all series |
| `refreshFitMenu()` | Rebuild the fit-mode dropdown (call after `register_fit_mode`) |
| `exportCsv()` | Open save dialog, write all series to CSV |
| `exportImage()` | Open save dialog, save canvas as PNG |
| `grabImage()` | Return `QPixmap` of the canvas without dialog |

All color arguments accept any string `QColor` understands: `"#e74c3c"`, `"red"`, `"rgb(200,100,50)"`.  
If `color` is omitted, a color is chosen automatically from the palette.

---

### Series items

```python
line.setData(xs=[...], ys=[...])
line.setVisible(True)
line.setLabel("My series")

scatter.setData(x=[...], y=[...])

fit.setModeKey("spline")     # change fit algorithm live
fit.setVisible(False)

hline = chart.addLine(y=100.0, color="#f39c12", dashed=True)
hline.setValue(150.0)
hline.setVisible(True)
```

---

### Built-in fit modes

| Key | Description | Min points |
|-----|-------------|------------|
| `linear_origin` | Linear through origin (y = kx) | 1 |
| `linear` | Ordinary linear regression | 2 |
| `poly2` | Polynomial degree 2 | 2 |
| `poly3` | Polynomial degree 3 | 2 |
| `poly4` | Polynomial degree 4 | 2 |
| `pchip` | Piecewise Cubic Hermite (monotone) | 2 |
| `spline` | Natural cubic spline | 2 |

---

### Custom fit modes

```python
from pyqt5_chart_widget import FitMode, register_fit_mode

def my_fit(x_pts, y_pts, x_eval):
    # x_pts / y_pts are pre-sorted, de-duplicated
    # return a list of float values the same length as x_eval
    k = sum(xi * yi for xi, yi in zip(x_pts, y_pts)) / sum(xi**2 for xi in x_pts)
    return [k * xi for xi in x_eval]

register_fit_mode(FitMode("my_fit", "My custom fit", my_fit, min_points=2))

# Then rebuild the dropdown if the chart is already visible:
chart.refreshFitMenu()
```

---

### Sidebar

```python
chart = ChartWidget(show_sidebar=True)
sb = chart.sidebar()

sb.addLabel("Controls")
sb.addSeparator()
sb.addButton("Toggle line", lambda: line.setVisible(not line.visible))
sb.addButton("Export CSV",  chart.exportCsv, tooltip="Save data")
sb.clear()
```

---

### Legend

```python
chart = ChartWidget(show_legend=True)
# or toggle at runtime:
chart.setLegendVisible(True)
```

Series are shown in the legend only when they have a `label` and are visible.

---

### Color palette

```python
from pyqt5_chart_widget import set_palette, reset_colors

set_palette(["#ff0000", "#00ff00", "#0000ff"])  # replaces built-in palette
reset_colors()                                    # restart auto-color index
```

---

### Internationalization

```python
from pyqt5_chart_widget import set_tr, update_strings

# Option 1 — hook into your own tr() function
set_tr(lambda key: my_app_translations.get(key, key))

# Option 2 — patch individual strings
update_strings({
    "chart_widget.btn_fit":  "⊞ Autofit",
    "chart_widget.btn_csv":  "💾 CSV",
})
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

## Module layout

```
pyqt5_chart_widget/
├── __init__.py      # public API re-exports
├── chart_widget.py  # ChartWidget — main widget & toolbar
├── canvas.py        # _PlotCanvas — rendering engine
├── items.py         # _LineItem, _ScatterItem, _FitItem, _InfLine
├── math_utils.py    # FitMode, built-in fitters, register_fit_mode, nice_ticks
├── sidebar.py       # SidebarLabel — optional left panel
├── palette.py       # auto-color palette helpers
└── i18n.py          # tr(), set_tr(), update_strings()
```

---

## Why not matplotlib / pyqtgraph?

Matplotlib embedded in Qt is slow to import, awkward to theme, and overkill for one calibration curve. Pyqtgraph is excellent but pulls in numpy as a hard dependency. This widget is ~900 lines total, has zero runtime dependencies beyond PyQt5, and does exactly what it says on the tin.

---

## License

MIT
