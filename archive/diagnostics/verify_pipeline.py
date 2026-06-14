"""Verify the saved pipeline matches training behavior.
Tests the pipeline on KNOWN files and checks it predicts sensibly."""
import pandas as pd
import numpy as np
from pathlib import Path
from liftdoctor.predict import diagnose, _load

DATA_DIR = Path("/Users/ibrahimshehab/Desktop/3W-main/dataset")
bundle = _load()

# Test one REAL file from each event folder, check the diagnosis
print("Testing pipeline on one known file per event type:\n")
print(f"{'Folder':>6} {'Expected':25} {'Predicted':40}")
print("-" * 75)

for event in range(10):
    folder = DATA_DIR / str(event)
    real_files = sorted(folder.glob("WELL-*.parquet"))
    if not real_files:
        real_files = sorted(folder.glob("*.parquet"))
    if not real_files:
        continue
    expected = bundle["event_names"][event]
    verdict, _, _ = diagnose(real_files[0], confidence=0.50)
    match = "✓" if expected.split()[0].lower() in verdict.lower() else "✗"
    print(f"{event:>6} {expected:25} {verdict:40} {match}")