##Plot training curves from runs/history.json, runs/curves.png
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
