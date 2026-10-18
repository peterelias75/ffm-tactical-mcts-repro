#!/usr/bin/env python3
import sys
import shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from chartlib import build
build(Path(__file__).resolve().parent.name)
shutil.copyfile(
    Path(__file__).with_name("chart.png"),
    Path(__file__).with_name("plot.png"),
)
