# Bank Transaction Fraud Detection System

A machine learning system for detecting fraudulent banking transactions. Uses a hybrid approach that combines rule-based checks with an Isolation Forest model.

## The Problem

Standard anomaly detection didn't work well here because probable fraud transactions made up a large portion of the dataset. These are organized attacks where the same device or IP is used across multiple accounts - basically device farms and botnets. A pure ML approach would miss these since they're not really "anomalies" in the traditional sense.

## The Solution

I built a two-stage system:

**Stage 1: Rule-based filtering**
- Checks if a device or IP address is linked to more than 3 unique accounts
- Catches the obvious organized crime stuff immediately
- This alone flags most of the fraud rings

**Stage 2: Isolation Forest model**
- Catches the sneaky stuff that passes the device checks
- Looks at transaction amounts, login attempts, balance ratios, etc.
- Detects things like account takeovers or someone draining their entire balance in one go

The API returns a risk score from 0-100% instead of just a binary fraud/not-fraud flag, which is more useful for downstream systems.

## Tech Stack

- Python 3.12+
- `uv` for package management
- Scikit-learn for the ML model
- FastAPI for the API
- Pandas/Numpy for data processing

## Setup

First, install `uv` if you don't have it:

```bash
# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then clone the repo and sync dependencies:

```bash
cd bank-fraud-detection
uv sync
```

This creates a virtual environment and installs everything automatically.

## Usage

### Training the Model

Run the notebook to do the EDA and train the model:

```bash
uv run jupyter notebook notebooks/analysis.ipynb
```

This will save the trained model and feature lookups to the `app/` directory.

### Running the API

Start the server:

```bash
uv run uvicorn app.main:app --reload
```

Then go to http://127.0.0.1:8000/docs for the interactive API docs.

### Example Requests

**Normal transaction:**
```json
{
  "AccountBalance": 8000.0,
  "Channel": "ATM",
  "CustomerOccupation": "Doctor",
  "DeviceID": "D000123",
  "IP_Address": "192.168.1.5",
  "LoginAttempts": 0,
  "TransactionAmount": 120.5
}
```
Should return a low risk score (~10%).

**Suspicious transaction:**
```json
{
  "AccountBalance": 1000.0,
  "Channel": "Online",
  "CustomerOccupation": "Student",
  "DeviceID": "D_New_User",
  "IP_Address": "10.0.0.1",
  "LoginAttempts": 3,
  "TransactionAmount": 950.0
}
```
This one should flag as high risk (>75%) - draining almost the entire balance with multiple failed logins.

## Project Structure

```
bank-fraud-detection/
├── app/
│   ├── main.py             # FastAPI app
│   ├── model.pkl           # Trained Isolation Forest
│   ├── scaler.pkl          # Feature scaler
│   ├── device_lookup.pkl   # Device history lookup
│   ├── ip_lookup.pkl       # IP history lookup
│   └── occ_lookup.pkl      # Occupation stats
├── data/
│   └── bank_transactions.csv
├── notebooks/
│   └── analysis.ipynb      # EDA and model training
├── pyproject.toml
├── uv.lock
└── README.md
```

## Some Interesting Findings

1. The dataset had a weird bug where `TransactionDate` was always before `PreviousTransactionDate` - probably a synthetic data generation issue. Had to work around this.

2. About 22% of online transactions share devices across multiple accounts, which is a strong indicator of botnet activity.

3. Some transactions have account balances that are way higher than others with the same occupation.
