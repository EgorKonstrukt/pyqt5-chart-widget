from setuptools import setup, find_packages

setup(
    name="pyqt5-chart-widget",
    version="1.0.0",
    description="Lightweight interactive chart widget for PyQt5",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Zarrakun",
    author_email="egormajndi@gmail.com",
    url="https://github.com/EgorKonstrukt/pyqt5-chart-widget",
    license="MIT",
    packages=find_packages(include=["pyqt5_chart_widget*"]),
    python_requires=">=3.8",
    install_requires=["PyQt5>=5.15"],
    keywords=["pyqt5", "chart", "plot", "widget", "gui"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: User Interfaces",
    ],
)