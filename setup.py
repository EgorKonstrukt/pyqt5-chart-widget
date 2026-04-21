import sys
import os
from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize

if len(sys.argv) == 1:
    sys.argv += ["build_ext", "--inplace"]

root = os.path.dirname(os.path.abspath(__file__))
pyx_path = os.path.join(root, "pyqt5_chart_widget", "_cy_utils.pyx")

setup(
    name="pyqt5-chart-widget",
    version="4.1.0",
    description="Lightweight interactive chart widget for PyQt5 with built-in approximation, multi-series, and extensible fit modes",
    long_description=open(os.path.join(root, "README.md"), encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Zarrakun",
    author_email="egormajndi@gmail.com",
    url="https://github.com/EgorKonstrukt/pyqt5-chart-widget",
    license="MIT",
    packages=find_packages(include=["pyqt5_chart_widget*"]),
    python_requires=">=3.8",
    install_requires=["PyQt5>=5.15"],
    keywords=["pyqt5", "chart", "plot", "widget", "gui", "approximation", "curve fitting"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: User Interfaces",
    ],
    ext_modules=cythonize(
        [Extension(
            "pyqt5_chart_widget._cy_utils",
            sources=[pyx_path],
            extra_compile_args=["-O2"],
        )],
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
        },
    ),
)