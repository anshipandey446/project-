import sys
import time
sys.path.append("src")

import pandas as pd
import streamlit as st
from detect_attacks import detect

# ---------- Page setup ----------
st.set_page_config(page_title="CAN-Sentinel", layout="wide")
st.title("🚗 CAN-Sentinel: CAN Intrusion Detection Dashboard")
st.caption("Dataset-based / simulated platform. It is not connected to a real vehicle.")

# ---------- Input: uploaded file or the default sample ----------
uploaded = st.sidebar.file_uploader("Upload CAN CSV file", type="csv")
source = uploaded if uploaded else "data/raw/can_raw.csv"
raw = pd.read_csv(source, dtype={"can_id": str, "data": str})


@st.cache_data
def run_detection(r):
    return detect(r)


df = run_detection(raw)
df["sec"] = (df["timestamp"] - df["timestamp"].min()).astype(int)

# ---------- Sidebar filters ----------
st.sidebar.header("Filters")
all_sev = ["High", "Medium", "Low", "None"]
all_status = sorted(df["status"].unique())
sev = st.sidebar.multiselect("Severity", all_sev, default=all_sev)
stat = st.sidebar.multiselect("Status", all_status, default=all_status)
view = df[df["severity"].isin(sev) & df["status"].isin(stat)]

# ---------- Replay simulation ----------
st.sidebar.header("Simulation")
if st.sidebar.button("▶ Replay traffic"):
    box = st.empty()
    bar = st.progress(0)
    last = int(df["sec"].max())
    for s in range(last + 1):
        now = df[df["sec"] == s]
        seen = df[df["sec"] <= s]
        text = (f"Time {s}s | messages so far: {len(seen):,} | "
                f"flagged so far: {int(seen['flagged'].sum()):,}")
        if now["flagged"].any():
            box.error("🚨 Suspicious traffic detected! " + text)
        else:
            box.success("✅ Traffic normal. " + text)
        bar.progress((s + 1) / (last + 1))
        time.sleep(0.15)

# ---------- Summary numbers ----------
total = len(df)
flagged = int(df["flagged"].sum())
c1, c2, c3, c4 = st.columns(4)
c1.metric("Messages analyzed", f"{total:,}")
c2.metric("Normal messages", f"{total - flagged:,}")
c3.metric("Flagged messages", f"{flagged:,}")
c4.metric("Attack categories", df.loc[df["ml_hit"], "prediction"].nunique())

# ---------- Alert banner ----------
if flagged > 0:
    top = df[df["flagged"]]["status"].value_counts().idxmax()
    st.error(f"🚨 ALERT: suspicious traffic detected! Most common: {top}")
else:
    st.success("✅ No suspicious traffic detected.")

# ---------- Charts ----------
left, right = st.columns(2)

left.subheader("Messages by status")
left.bar_chart(df["status"].value_counts())

right.subheader("Attack timeline (flagged messages per second)")
timeline = df[df["flagged"]].groupby(["sec", "status"]).size().unstack(fill_value=0)
if timeline.empty:
    right.info("No attacks to plot.")
else:
    right.area_chart(timeline)

st.subheader("CAN ID activity (top 15 IDs)")
st.bar_chart(df["can_id_hex"].value_counts().head(15))

# ---------- Prediction table ----------
cols = ["timestamp", "can_id_hex", "dlc", "data", "status", "severity"]
st.subheader("Predictions (filtered, first 1,000 rows)")
st.dataframe(view[cols].head(1000), use_container_width=True)

# ---------- Alert log and report ----------
alerts = df[df["flagged"]][cols]
st.subheader("Alert log (first 200 alerts)")
st.dataframe(alerts.head(200), use_container_width=True)

st.download_button("⬇ Download detection report (CSV)",
                   alerts.to_csv(index=False).encode(),
                   "detection_report.csv", "text/csv")

# ---------- Accuracy check if the file has labels ----------
if "label" in df.columns:
    agree = (df["prediction"] == df["label"]).mean() * 100
    st.sidebar.info(f"Label column found. Model agrees with labels on {agree:.2f}% of messages.")