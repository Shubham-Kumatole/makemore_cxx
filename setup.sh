#!/bin/bash
set -e

echo "Installing Python dependencies..."
python3 -m pip install matplotlib numpy --break-system-packages

echo "Installing Tkinter for interactive backend..."
brew install python-tk

echo "Downloading matplotlib-cpp header..."
curl -L https://raw.githubusercontent.com/lava/matplotlib-cpp/master/matplotlibcpp.h \
    -o include/matplotlibcpp.h

echo "Done."