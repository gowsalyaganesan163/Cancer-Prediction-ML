# Terminal + Streamlit versions

The original terminal/GUI entry point is preserved. The new browser UI is `streamlit_app.py`.

## Run the original project
Use the original command shown in its original project README (if available), or run its `main.py`/script from the project directory. Existing terminal source files were not replaced.

## Run the Streamlit UI
1. Open a terminal in this project directory.
2. Create and activate a virtual environment (recommended).
3. Install UI dependencies:
   `pip install -r requirements_streamlit.txt`
4. Launch:
   `streamlit run streamlit_app.py`

The UI runs locally in your browser. CSV datasets are uploaded through the interface unless an included CSV is used by the app.

Note: The Cancer Prediction ZIPs were identical copies; only one copy is included in this converted bundle.
