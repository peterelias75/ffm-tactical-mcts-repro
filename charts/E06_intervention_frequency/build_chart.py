#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from chartlib import build
build(Path(__file__).resolve().parent.name)
