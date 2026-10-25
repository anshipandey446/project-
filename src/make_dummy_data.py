import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

DURATION = 60.0   # we simulate 60 seconds of car traffic

# Normal messages: CAN ID -> how often it repeats (in seconds)
NORMAL = {0x1A0: 0.010, 0x2B0: 0.020, 0x316: 0.010, 0x153: 0.020, 0x260: 0.050}

rows = []

def hexdata(byte_list):
    return " ".join(f"{b:02X}" for b in byte_list)

# ---------- 1. Normal traffic ----------
for cid, period in NORMAL.items():
    t = rng.uniform(0, period)
    while t < DURATION:
        data = [int(rng.integers(50, 100))] + [0] * 7
        rows.append((t, f"{cid:04X}", 8, hexdata(data), "Normal"))
        t += period + rng.normal(0, period * 0.05)

# ---------- 2. DoS attacks: ID 0000, very fast, 1 second each ----------
for start in (8, 18, 28, 38, 48):
    t = float(start)
    while t < start + 1.0:
        rows.append((t, "0000", 8, hexdata([0] * 8), "DoS"))
        t += 0.0003

# ---------- 3. Fuzzy attacks: random IDs and random data, 0.5 second each ----------
for start in (13, 23, 33, 43, 53):
    t = float(start)
    while t < start + 0.5:
        rows.append((t, f"{int(rng.integers(0, 0x800)):04X}", 8,
                     hexdata(rng.integers(0, 256, 8)), "Fuzzy"))
        t += 0.0005

# ---------- 4. Put everything in one table and save ----------
df = pd.DataFrame(rows, columns=["timestamp", "can_id", "dlc", "data", "label"])
df = df.sort_values("timestamp").reset_index(drop=True)
df.to_csv("data/raw/can_raw.csv", index=False)

print("File saved: data/raw/can_raw.csv")
print("Total messages:", len(df))
print(df["label"].value_counts())