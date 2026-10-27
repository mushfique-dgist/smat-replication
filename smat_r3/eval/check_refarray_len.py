"""Diagnostic: compares the reference file's data_length metadata with the actual array lengths."""
import numpy as np
import os
REF = os.path.join(os.path.dirname(__file__), "../../ext/myoassist_3b945c7/rl_train/reference_data/short_reference_gait.npz")
d = np.load(REF, allow_pickle=True)
meta = d["metadata"].item()
series = d["series_data"].item()
print("metadata:", meta)
for k in ["q_hip_flexion_r", "q_pelvis_ty", "q_pelvis_tx"]:
    v = series[k]
    print(k, "type=", type(v), "len=", len(v))
print("num series_data keys:", len(series.keys()))
