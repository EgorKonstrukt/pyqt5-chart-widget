from __future__ import annotations
from typing import Callable, Dict, Optional

_STRINGS: Dict[str, str] = {
    "chart_widget.btn_fit":                 "Fit",
    "chart_widget.btn_fit_tip":             "Auto-fit view to data (double-click on chart)",
    "chart_widget.btn_csv":                 "CSV",
    "chart_widget.btn_csv_tip":             "Export data to CSV",
    "chart_widget.btn_img":                 "Image",
    "chart_widget.btn_img_tip":             "Export chart as PNG image",
    "chart_widget.btn_analytics":           "Stats",
    "chart_widget.btn_analytics_tip":       "Show / hide analytics panel",
    "chart_widget.btn_fit_mode":            "Fit",
    "chart_widget.btn_fit_mode_tip":        "Select approximation mode",
    "chart_widget.btn_autofit_toggle":      "Auto-fit",
    "chart_widget.btn_autofit_toggle_tip":  "Auto-fit view when data changes",
    "chart_widget.btn_latest":              "Latest",
    "chart_widget.btn_latest_tip":          "Show latest point values on axis rulers",
    "chart_widget.csv_title":               "Export to CSV",
    "chart_widget.csv_filter":              "CSV files (*.csv);;All files (*)",
    "chart_widget.img_title":               "Export image",
    "chart_widget.img_filter":              "PNG images (*.png);;All files (*)",
    "chart_widget.tooltip_x":              "X",
    "chart_widget.tooltip_y":              "Y",
    "chart_widget.analytics_line":          "Line {n}",
    "chart_widget.analytics_scatter":       "Scatter {n}",
    "chart_widget.analytics_fit":           "Fit {n}",
    "chart_widget.analytics_n":             "n",
    "chart_widget.analytics_xmin":          "x min",
    "chart_widget.analytics_xmax":          "x max",
    "chart_widget.analytics_ymin":          "y min",
    "chart_widget.analytics_ymax":          "y max",
    "chart_widget.analytics_mean":          "mean y",
    "chart_widget.analytics_std":           "std y",
    "chart_widget.legend_label":            "Series {n}",
}

_custom_tr: Optional[Callable[[str], str]] = None


def set_tr(fn: Callable[[str], str]):
    global _custom_tr
    _custom_tr = fn


def tr(key: str, **kwargs) -> str:
    if _custom_tr is not None:
        try:
            text = _custom_tr(key)
            return text.format(**kwargs) if kwargs else text
        except Exception:
            pass
    text = _STRINGS.get(key, key.split(".")[-1])
    return text.format(**kwargs) if kwargs else text


def update_strings(mapping: Dict[str, str]):
    _STRINGS.update(mapping)