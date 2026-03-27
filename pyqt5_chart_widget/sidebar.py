from __future__ import annotations

from typing import Callable

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QLabel,
                             QSizePolicy, QToolButton, QFrame)


class SidebarButton(QToolButton):
    def __init__(self, text: str, parent=None):
        super().__init__(parent)
        self.setText(text)
        self.setFixedHeight(26)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)


class SidebarLabel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(4, 4, 4, 4)
        self._layout.setSpacing(4)
        self._layout.addStretch()
        self.setFixedWidth(110)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        line = QFrame(self)
        line.setFrameShape(QFrame.Shape.VLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)

    def addButton(self, text: str, callback: Callable, tooltip: str = "") -> SidebarButton:
        btn = SidebarButton(text, self)
        if tooltip:
            btn.setToolTip(tooltip)
        btn.clicked.connect(callback)
        self._layout.insertWidget(self._layout.count() - 1, btn)
        return btn

    def addLabel(self, text: str) -> QLabel:
        lbl = QLabel(text, self)
        lbl.setWordWrap(True)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._layout.insertWidget(self._layout.count() - 1, lbl)
        return lbl

    def addSeparator(self):
        line = QFrame(self)
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        self._layout.insertWidget(self._layout.count() - 1, line)

    def clear(self):
        while self._layout.count() > 1:
            item = self._layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
