# CAN-Sentinel: CAN Bus Intrusion Detection (Student 2 module)

A dataset-based, simulated CAN intrusion detection platform. It is not connected to a real vehicle.

## What it does
- Trains and compares a Decision Tree and a Random Forest on CAN traffic features
- Detects and classifies Normal, DoS and Fuzzy traffic
- Combines the ML model with rule-based checks (blocklisted ID, too-fast repetition)
- Assigns each message a status and a severity (High / Medium / Low)
- Streamlit dashboard with charts, filters, alert banner, alert log and CSV report
- Isolation Forest anomaly layer (trained on normal traffic only) to flag unusual traffic

## Project structure
```
src/make_dummy_data.py     simulated CAN traffic generator
src/feature_extraction.py  turns raw messages into model features
src/train_model.py         trains, compares and saves the model
src/detect_attacks.py      ML + rule-based detection and severity
src/export_report.py       exports a sample detection report
app.py                     Streamlit dashboard
reports/                   evaluation report, confusion matrix, sample report
```

## How to run
```
pip install -r requirements.txt
python src/make_dummy_data.py
python src/train_model.py
streamlit run app.py
```

## Results so far
- Random Forest was selected over the Decision Tree (higher macro F1).
- Scores on the simulated data are near-perfect because the simulated attacks are easy to separate. They will be re-evaluated on the real dataset.

## Current status and limitations
- Integration with the Module 1 (data collection and preprocessing) dataset is pending.
- This is a dataset-based simulation, not a real-time vehicle security system.
- Replay attacks are not covered yet.
- - Planned next: evaluation on the real dataset