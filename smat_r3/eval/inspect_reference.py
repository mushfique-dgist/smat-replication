"""Diagnostic: prints the reference file's metadata and series keys."""
import numpy as np
import os

PATH = os.path.join(os.path.dirname(__file__), "../../ext/myoassist_3b945c7/rl_train/reference_data/short_reference_gait.npz")
d = np.load(PATH, allow_pickle=True)
meta = d["metadata"].item()
series = d["series_data"].item()

print("=== metadata ===")
for k, v in meta.items():
    print(k, ":", v)

print("=== series_data keys ===")
keys = sorted(series.keys())
print(len(keys), "keys")
for k in keys:
    arr = series[k]
    print(k, getattr(arr, "shape", type(arr)))
