# Phase 3 checklist

## Files
- app.py
- requirements.txt
- insurance.csv
- phase3.pdf

## Start MLflow server
python -m mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns

## Start Streamlit app
streamlit run app.py

## What to screenshot for phase3.pdf
1. Sidebar / user input section
2. Prediction output
3. Key insights chart
4. Optional screenshot showing confidence interval

## What to explain in phase3.pdf
- The app loads the registered MLflow model locally
- User inputs are transformed into a pandas DataFrame
- The model predicts insurance charges
- A local OLS model provides the 95% confidence interval
