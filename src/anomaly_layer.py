import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from feature_extraction import build_features, FEATURES

RAW = "data/raw/can_raw.csv"
MODEL_OUT = "models/anomaly_model.pkl"

# 1. Load data and build features
df = build_features(pd.read_csv(RAW, dtype={"can_id": str, "data": str}))
X, y = df[FEATURES], df["label"]

# 2. Split, then learn ONLY from normal traffic in the training part
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
normal_train = X_train[y_train == "Normal"]

iso = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
iso.fit(normal_train)

# 3. Test on unseen traffic: -1 means "anomaly", 1 means "looks normal"
flagged = iso.predict(X_test) == -1
result = pd.DataFrame({"label": y_test.values, "flagged_as_anomaly": flagged})
rate = result.groupby("label")["flagged_as_anomaly"].mean().mul(100).round(2)

lines = [
    "ISOLATION FOREST ANOMALY LAYER (trained on Normal traffic only)",
    "",
    "% of each real class flagged as anomaly:",
    rate.to_string(),
    "",
    "Reading: DoS and Fuzzy should be high (attacks caught).",
    "Normal should be low (a high Normal value means false alarms).",
]
text = "\n".join(lines)
print(text)
open("reports/anomaly_report.txt", "w").write(text)

joblib.dump({"model": iso, "features": FEATURES}, MODEL_OUT)
print("\nSaved model to", MODEL_OUT)