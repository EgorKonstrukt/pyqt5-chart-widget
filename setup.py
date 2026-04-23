import sys
import os
import platform
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext

if len(sys.argv) == 1:
    sys.argv += ["build_ext", "--inplace"]

root = os.path.dirname(os.path.abspath(__file__))
pyx_path = os.path.join(root, "pyqt5_chart_widget", "_cy_utils.pyx")


def get_compile_args():
    if platform.system() == "Windows":
        return ["/O2", "/W0"]
    return ["-O2"]


def get_link_args():
    if platform.system() == "Windows":
        return []
    return []


class BuildExtFixed(build_ext):
    def build_extensions(self):
        compiler = self.compiler.compiler_type
        for ext in self.extensions:
            if compiler == "msvc":
                ext.extra_compile_args = ["/O2", "/W0"]
                ext.extra_link_args = []
            else:
                ext.extra_compile_args = ["-O2"]
                ext.extra_link_args = []
        super().build_extensions()

    def finalize_options(self):
        super().finalize_options()
        if platform.system() == "Windows":
            import tempfile
            short_tmp = os.path.join(os.path.dirname(root), "_cy_build")
            os.makedirs(short_tmp, exist_ok=True)
            self.build_temp = short_tmp


try:
    from Cython.Build import cythonize

    ext_modules = cythonize(
        [Extension(
            "pyqt5_chart_widget._cy_utils",
            sources=[pyx_path],
            extra_compile_args=get_compile_args(),
            extra_link_args=get_link_args(),
        )],
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
        },
    )
except Exception:
    ext_modules = []

setup(
    name="pyqt5-chart-widget",
    version="4.2.1",
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
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtFixed},
)