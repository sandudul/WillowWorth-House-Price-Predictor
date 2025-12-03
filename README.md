# House Price Prediction

Simple project that analyzes a housing dataset and provides a Streamlit UI to predict median house value.

## What this repo contains
- `1.Data Analysis.ipynb` — exploratory notebook with preprocessing and model experiments.
- `housing.csv` — (expected) dataset used for training and exploration.
- `streamlit_app.py` — Streamlit application for interactive predictions.
- `requirements.txt` — Python packages needed to run the app.

## Quickstart (Windows PowerShell)
1. Open PowerShell and change to the project folder:

```powershell
cd "C:\Users\sandu\OneDrive\Desktop\House Price Prediction"
```

2. (Optional) Create and activate a virtual environment:

```powershell
python -m venv .\.venv
. .\.venv\Scripts\Activate.ps1
```

3. Install dependencies:

```powershell
pip install -r .\requirements.txt
```

4. Run the Streamlit app:

```powershell
streamlit run .\streamlit_app.py
```

5. Open the app in your browser at http://localhost:8501.

## Notes
- The app trains simple models at startup using `housing.csv`. For faster startup in the future, consider training models once and saving them to disk (joblib / pickle) and then loading them in `streamlit_app.py`.
- This project is intended for demonstration and learning — do not use it in production without further validation.

## Troubleshooting
- If Streamlit fails to start, ensure the virtual environment is activated and dependencies are installed.
- If the dataset file `housing.csv` is missing, place it in the project root.

If you want, I can modify the app to save/load a trained model to speed startup, or add a sample `saved_model.pkl` flow. Which would you like next?
