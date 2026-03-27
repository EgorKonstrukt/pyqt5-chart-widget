from .chart_widget import ChartWidget
from .items import _LineItem, _ScatterItem, _FitItem, _InfLine
from .math_utils import FitMode, register_fit_mode, get_fit_modes, get_fit_mode
from .sidebar import SidebarLabel, SidebarButton
from .palette import set_palette, reset_colors
from .i18n import tr, set_tr, update_strings

__all__ = [
    "ChartWidget",
    "_LineItem",
    "_ScatterItem",
    "_FitItem",
    "_InfLine",
    "FitMode",
    "register_fit_mode",
    "get_fit_modes",
    "get_fit_mode",
    "SidebarLabel",
    "SidebarButton",
    "set_palette",
    "reset_colors",
    "tr",
    "set_tr",
    "update_strings",
]
__version__ = "2.0.0"
