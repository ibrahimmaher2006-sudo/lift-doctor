"""Show EVERY window's prediction for one fault file, to see if the
failure is detected later in the file even when the majority is Normal."""
from pathlib import Path
from liftdoctor.predict import predict_file

# A BSW-increase file (folder 1) — verify said this wrongly aggregates to "Normal"
folder = Path("/Users/ibrahimshehab/Desktop/3W-main/dataset/1")
f = sorted(folder.glob("WELL-*.parquet"))[0]

results = predict_file(f)
print(f"File: {f.name}  ({len(results)} windows)\n")

# Count what's predicted across the whole file
from collections import Counter
counts = Counter(name for _, name, _ in results)
print("Label counts across file:", dict(counts), "\n")

# Show the sequence — do later windows differ from early ones?
for start, name, conf in results:
    print(f"  t={start:5d}s : {name:25s} ({conf})")