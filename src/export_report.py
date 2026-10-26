import pandas as pd
from detect_attacks import detect

# Run the detector on the CAN data
raw = pd.read_csv("data/raw/can_raw.csv", dtype={"can_id": str, "data": str})
out = detect(raw)

# Keep up to 200 alerts of each type, so the sample shows DoS and Fuzzy
cols = ["timestamp", "can_id_hex", "dlc", "data", "status", "severity"]
sample = (out[out["flagged"]]
          .groupby("status").head(200)
          .sort_values("timestamp")[cols])

sample.to_csv("reports/sample_detection_report.csv", index=False)
print("Saved reports/sample_detection_report.csv with", len(sample), "alerts")

