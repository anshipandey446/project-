import joblib
import pandas as pd
from feature_extraction import build_features

MODEL_PATH = "models/can_ids_model.pkl"
NORMAL_LABEL = "Normal"

# ---- Rule layer settings ----
BLOCKLIST = {0x000}      # Rule 1: this ID should never appear on the bus
MIN_ID_GAP = 0.0005      # Rule 2: same ID repeating faster than 0.5 ms

_bundle = None


def load_model():
    """Load the saved model once and reuse it."""
    global _bundle
    if _bundle is None:
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def detect(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Take raw CAN messages and return them with status and severity added."""
    bundle = load_model()
    df = build_features(raw_df)

    # 1. ML layer: what does the model predict?
    df["prediction"] = bundle["model"].predict(df[bundle["features"]])
    df["ml_hit"] = df["prediction"] != NORMAL_LABEL

    # 2. Rule layer: did a fixed rule fire?
    df["rule_hit"] = df["can_id"].isin(BLOCKLIST) | (df["id_delta_t"] < MIN_ID_GAP)

    # 3. Readable status
    df["status"] = "Normal"
    df.loc[df["rule_hit"], "status"] = "SUSPICIOUS (rule)"
    df.loc[df["ml_hit"], "status"] = "ATTACK: " + df.loc[df["ml_hit"], "prediction"].astype(str)

    # 4. Severity rule
    #    Low    = only a rule fired
    #    Medium = only the ML model flagged it
    #    High   = ML and rule agree, OR it is a DoS attack
    df["severity"] = "None"
    df.loc[df["rule_hit"], "severity"] = "Low"
    df.loc[df["ml_hit"], "severity"] = "Medium"
    df.loc[df["ml_hit"] & df["rule_hit"], "severity"] = "High"
    df.loc[df["prediction"] == "DoS", "severity"] = "High"

    df["flagged"] = df["status"] != "Normal"
    df["can_id_hex"] = df["can_id"].map(lambda v: f"0x{v:03X}")
    return df


if __name__ == "__main__":
    raw = pd.read_csv("data/raw/can_raw.csv", dtype={"can_id": str, "data": str})
    out = detect(raw)

    print("STATUS COUNTS:")
    print(out["status"].value_counts(), "\n")
    print("SEVERITY COUNTS:")
    print(out["severity"].value_counts(), "\n")

    print("FIRST 10 ALERTS:")
    cols = ["timestamp", "can_id_hex", "status", "severity"]
    print(out[out["flagged"]][cols].head(10).to_string(index=False))

    if "label" in out.columns:
        print("\nREAL LABEL vs OUR STATUS:")
        print(pd.crosstab(out["label"], out["status"]))