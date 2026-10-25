import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report)
from feature_extraction import build_features, FEATURES

RAW = "data/raw/can_raw.csv"
MODEL_OUT = "models/can_ids_model.pkl"

# 1. Load the data and turn it into features
df = build_features(pd.read_csv(RAW, dtype={"can_id": str}))

# 2. Separate the inputs (X) from the correct answers (y)
assert "label" not in FEATURES, "Leakage: label used as a feature!"
X, y = df[FEATURES], df["label"]

report = ["CLASS COUNTS (imbalance check):", y.value_counts().to_string(), ""]

# 3. Split: 80% to learn from, 20% hidden for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# 4. The two models we will compare
models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=10, class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                            random_state=42, n_jobs=-1),
}

# 5. Train each model and test it on the hidden 20%
results = {}
for name, m in models.items():
    m.fit(X_train, y_train)
    pred = m.predict(X_test)
    p, r, f1, _ = precision_recall_fscore_support(y_test, pred, average="macro", zero_division=0)
    results[name] = {"model": m, "pred": pred, "f1": f1}
    report += [f"=== {name} ===",
               f"Accuracy {accuracy_score(y_test, pred):.4f} | Macro Precision {p:.4f} | "
               f"Macro Recall {r:.4f} | Macro F1 {f1:.4f}",
               classification_report(y_test, pred, zero_division=0)]

# 6. Pick the winner and draw its confusion matrix
best_name = max(results, key=lambda k: results[k]["f1"])
best = results[best_name]
report.append(f"SELECTED MODEL: {best_name} (highest macro F1)")

labels = sorted(y.unique())
cm = confusion_matrix(y_test, best["pred"], labels=labels)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix - {best_name}")
plt.savefig("reports/graphs/confusion_matrix.png", bbox_inches="tight")

# 7. Leakage check: which features did the winner rely on?
imp = pd.Series(best["model"].feature_importances_, index=FEATURES).sort_values(ascending=False)
report += ["", "FEATURE IMPORTANCE (if one is ~1.0, investigate leakage):", imp.round(3).to_string()]

# 8. Save the winner so the dashboard can use it later
joblib.dump({"model": best["model"], "features": FEATURES}, MODEL_OUT)

text = "\n".join(report)
open("reports/evaluation_report.txt", "w").write(text)
print(text)
print("\nSaved model to", MODEL_OUT)