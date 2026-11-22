# Breast Cancer Prediction — Logistic Regression

## Overview
Educational classification demo using breast-cancer features and a diagnosis target. This is not a medical tool.

## Streamlit UI setup

**Prerequisites:** Python installed.

Open a terminal in this project directory and create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

**PowerShell**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Command Prompt (CMD)**
```bat
.venv\Scripts\activate.bat
```

Install the Streamlit app dependencies:

```bash
pip install -r requirements_streamlit.txt
```

Start the app:

```bash
streamlit run streamlit_app.py
```

Streamlit prints the local address (typically `http://localhost:8501`) and opens the UI in your browser.

## Original terminal version
The original source files are retained. Run the terminal entry point from this folder:

```bash
python Cancer_prediction_code.py
```

If required, install the original project's dependencies from its original requirements file. `requirements_streamlit.txt` covers the UI version.

## Features
CSV upload and preview; class distribution; Logistic Regression; accuracy/precision/recall/F1; confusion matrix and classification report.

## Dataset requirements
Upload a CSV with diagnosis values B (Benign) and M (Malignant). Optional id and Unnamed: 32 columns are dropped; other feature columns should be numeric.

## Notes
- Run commands from the project directory to ensure relative paths work.
- Keep the original scripts and data files in place.
- This Streamlit app runs locally; it is not automatically deployed online.
