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
win.setWindowTitle("ChartWidget demo")
win.show()
sys.exit(app.exec_())