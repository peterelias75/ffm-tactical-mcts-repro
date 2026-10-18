#!/usr/bin/env python3
import sys
import shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from chartlib import build
build(Path(__file__).resolve().parent.name)
chart = Path(__file__).with_name("chart.png")
shutil.copyfile(chart, Path(__file__).with_name("plot.png"))
chart.unlink()
