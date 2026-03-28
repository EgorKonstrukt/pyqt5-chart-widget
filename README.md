# pyqt5-chart-widget

Zero dependencies beyond PyQt5. Drop the package in, plot your data. No matplotlib, no numpy — ~1000 lines total.

Originally built for an industrial powder feeder control application where a calibration curve needed to render fast, look clean, and not require importing half of scipy.

## Install

```bash
pip install pyqt5-chart-widget
```

Python >= 3.8, PyQt5 >= 5.15.

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

## ChartWidget

```python
ChartWidget(
    parent=None,
    show_toolbar:  bool = True,
    show_legend:   bool = False,
    show_sidebar:  bool = False,
    font:          QFont | None = None,
    threaded_fit:  bool = False,
)
```

`threaded_fit=True` moves fit computation off the main thread. Useful for large datasets or slow spline fits. Leave it `False` if you can't create additional threads — some embedded runtimes and multiprocessing workers don't allow it.

| Method | Description |
|--------|-------------|
| `plot(color, width, label, dashed)` | Add a line series → `_LineItem` |
| `addScatter(size, color, label)` | Add a scatter series → `_ScatterItem` |
| `addFit(source, mode_key, color, width, dashed, label)` | Add a fit curve for a series → `_FitItem` |
| `addLine(y=, x=, color, width, dashed)` | Add an infinite reference line → `_InfLine` |
| `setLabel(side, text)` | Set axis label — `"left"` or `"bottom"` |
| `autofit()` | Fit view to current data bounds |
| `setAutofitEnabled(bool)` | Enable/disable auto-fit when data changes (default: `True`) |
| `setLatestPointVisible(bool)` | Show/hide latest-point badges on axis rulers |
| `setThreadedFit(bool)` | Switch threaded fit computation on/off at runtime |
| `setToolbarVisible(visible)` | Show/hide the toolbar |
| `setSidebarVisible(visible)` | Show/hide the sidebar |
| `setLegendVisible(visible)` | Show/hide the legend overlay |
| `sidebar()` | Return the `SidebarLabel` instance (or `None`) |
| `removeItem(item)` | Remove any series or line |
| `clearAll()` | Remove all series |
| `refreshFitMenu()` | Rebuild the fit-mode dropdown (call after `register_fit_mode`) |
| `exportCsv()` | Open save dialog, write all series to CSV |
| `exportImage()` | Open save dialog, save canvas as PNG |
| `grabImage()` | Return `QPixmap` of the canvas without a dialog |

Color arguments accept any string `QColor` understands: `"#e74c3c"`, `"red"`, `"rgb(200,100,50)"`. Omit `color` to get one from the auto-palette.

---

## Series

```python
line = chart.plot(label="Sensor A")
line.setData(xs=[...], ys=[...])
line.setVisible(True)
line.setRawVisible(False)   # hide raw data points; keep the fit curve visible
line.setLabel("New label")

scatter = chart.addScatter(size=10, label="Samples")
scatter.setData(x=[...], y=[...])
scatter.setRawVisible(False)

fit = chart.addFit(line, mode_key="spline", label="Spline fit")
fit.setModeKey("poly3")     # change algorithm live
fit.setVisible(False)

hline = chart.addLine(y=100.0, color="#f39c12", dashed=True)
hline.setValue(150.0)
hline.setVisible(True)
```

`setRawVisible(False)` hides the raw line or scatter while leaving any associated fit curve untouched. Use this when you only want to show the approximation.

---

## Getting fit data out of the chart

```python
fit = chart.addFit(line, mode_key="poly3")
line.setData(xs=[...], ys=[...])

xs, ys = fit.getData()
# Returns a 400-point dense evaluation across the source data's x range.
# Both xs and ys are plain Python lists of floats.

xs, ys = fit.getData(x_lo=0.0, x_hi=500.0, n_pts=1000)
# Supply an explicit range and resolution when needed.
```

`getData()` always runs synchronously, regardless of whether `threaded_fit` is on. It's safe to call from any code path that needs the numbers.

---

## Built-in fit modes

| Key | Method | Min points |
|-----|--------|------------|
| `linear_origin` | y = kx (through origin) | 1 |
| `linear` | Ordinary least-squares | 2 |
| `poly2` | Polynomial degree 2 | 2 |
| `poly3` | Polynomial degree 3 | 2 |
| `poly4` | Polynomial degree 4 | 2 |
| `pchip` | Piecewise Cubic Hermite (monotone) | 2 |
| `spline` | Natural cubic spline | 2 |

---

## Custom fit modes

```python
from pyqt5_chart_widget import FitMode, register_fit_mode

def my_fit(x_pts, y_pts, x_eval):
    # x_pts / y_pts are pre-sorted and de-duplicated before this is called.
    # Return a list of floats with the same length as x_eval.
    k = sum(xi * yi for xi, yi in zip(x_pts, y_pts)) / sum(xi**2 for xi in x_pts)
    return [k * xi for xi in x_eval]

register_fit_mode(FitMode("my_fit", "My custom fit", my_fit, min_points=2))

# If the chart is already on screen, rebuild the dropdown:
chart.refreshFitMenu()
```

---

## Threaded fit computation

For large datasets or expensive spline fits, pass `threaded_fit=True`:

```python
chart = ChartWidget(threaded_fit=True)
```

Or flip it at runtime:

```python
chart.setThreadedFit(True)
```

While a fit is computing, the canvas shows the previous cached result. The update arrives asynchronously and triggers a repaint automatically. If you zoom or pan quickly, the pending computation is queued and runs once the in-flight worker finishes.

Leave `threaded_fit=False` (the default) if creating threads is not allowed in your environment — some embedded runtimes, multiprocessing child processes, and certain test frameworks don't support it.

---

## Auto-fit toggle

By default, the view refits whenever you push new data via `setData`. Disable it to manage the viewport yourself:

```python
chart.setAutofitEnabled(False)
chart.autofit()  # still works on demand
```

The toolbar has an **Auto-fit** toggle button for this. Double-clicking the chart always triggers a fit regardless of the setting.

---

## Latest-point markers

```python
chart.setLatestPointVisible(True)
```

Or use the **Latest** button in the toolbar. Each visible series gets a small colored badge on the X and Y axis rulers showing the last value in the series. Handy when streaming live data and you need to read the current value without hovering.

---

## Crosshair and hover

Hover over the plot area to get a crosshair, a snap dot on the nearest point, and an X/Y tooltip. The snap works on:

- Individual scatter points
- Interpolated positions along line segments (not just the data points)
- Interpolated positions on fit curves

A tangent line is drawn at the snap point for line series and fit curves.

---

## Sidebar

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

## Legend

```python
chart = ChartWidget(show_legend=True)
# or later:
chart.setLegendVisible(True)
```

Only series that have a `label` and are visible appear in the legend.

---

## Color palette

```python
from pyqt5_chart_widget import set_palette, reset_colors

set_palette(["#ff0000", "#00ff00", "#0000ff"])  # replace the built-in palette
reset_colors()                                    # restart the auto-color index
```

---

## Internationalization

```python
from pyqt5_chart_widget import set_tr, update_strings

# Hook into your own translation system:
set_tr(lambda key: my_translations.get(key, key))

# Or patch individual strings:
update_strings({
    "chart_widget.btn_fit":      "Fit view",
    "chart_widget.btn_csv":      "Save CSV",
    "chart_widget.btn_autofit_toggle": "Live fit",
    "chart_widget.btn_latest":   "Now",
})
```

---

## Interaction

| Action | Result |
|--------|--------|
| Scroll wheel | Zoom in/out centered on cursor |
| Left drag | Pan |
| Double-click | Reset view to data bounds |
| Hover | Crosshair + snap to nearest point/curve + tangent |

---

## Module layout

```
pyqt5_chart_widget/
├── __init__.py       — public API re-exports
├── chart_widget.py   — ChartWidget, toolbar
├── canvas.py         — rendering, interaction, hover
├── items.py          — _LineItem, _ScatterItem, _FitItem, _InfLine
├── math_utils.py     — FitMode, built-in fitters, register_fit_mode, nice_ticks
├── sidebar.py        — SidebarLabel
├── palette.py        — auto-color palette
└── i18n.py           — tr(), set_tr(), update_strings()
```

---

## Why not matplotlib or pyqtgraph?

Matplotlib embedded in Qt is slow to start, awkward to theme, and brings in a lot of machinery for one calibration curve. Pyqtgraph is excellent but requires numpy as a hard dependency. This widget is ~1000 lines, has zero runtime dependencies beyond PyQt5, and does exactly what the name says.

---

## License

MIT