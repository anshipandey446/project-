import pandas as pd

# The list of numbers the model will learn from
FEATURES = ["can_id", "dlc", "delta_t", "id_delta_t", "id_rate", "msg_rate"] + \
           [f"b{i}" for i in range(8)]


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.sort_values("timestamp").reset_index(drop=True).copy()

    # Convert the hex ID (like 01A0) into a normal number
    df["can_id"] = df["can_id"].apply(lambda v: int(str(v), 16))

    # Time gap since the previous message (any ID)
    df["delta_t"] = df["timestamp"].diff().fillna(0)

    # Time gap since the last message with the SAME ID
    df["id_delta_t"] = df.groupby("can_id")["timestamp"].diff().fillna(1.0)

    # Count messages in each 0.1-second window
    bucket = (df["timestamp"] // 0.1).astype(int)
    df["id_rate"] = df.groupby([bucket, df["can_id"]])["timestamp"].transform("count")
    df["msg_rate"] = df.groupby(bucket)["timestamp"].transform("count")

    # Turn the 8 payload bytes into 8 number columns: b0 ... b7
    hexes = df["data"].astype(str).str.replace(" ", "").str.ljust(16, "0")
    for i in range(8):
        df[f"b{i}"] = hexes.str[2 * i:2 * i + 2].apply(lambda s: int(s, 16))

    return df