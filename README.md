# Medical Insurance Charges Prediction - Phase 1

## Project Overview
This project was developed for **Phase 1** of the MLOps assignment.  
The objective is to build a complete classical machine learning workflow using **scikit-learn** and deploy the trained model in a **static client-side web application**.

The project uses a **Linear Regression** model to predict **medical insurance charges** based on personal and health-related features.

## Dataset
The dataset used in this project is the **Medical Insurance Cost Dataset**.

**Source:**  
https://www.kaggle.com/datasets/mosapabdelghany/medical-insurance-cost-dataset?resource=download

### Features
- `age`: age of the insured person
- `sex`: gender of the insured person
- `bmi`: body mass index
- `children`: number of dependents covered by insurance
- `smoker`: smoking status
- `region`: residential region in the United States
- `charges`: individual medical insurance cost (**target**)

## Project Structure
```text
DSAI_Insurance/
│── insurance.csv
│── phase1.ipynb
└── docs/
    ├── index.html
    ├── styles.css
    ├── predict.js
    ├── model.json
    ├── metrics.json
    ├── residuals.png
    └── pred_vs_actual.png
```

## Phase 1 Workflow
The notebook `phase1.ipynb` contains the complete Phase 1 pipeline:

1. **Data loading and inspection**
2. **Exploratory Data Analysis (EDA)**
3. **Data preprocessing**
   - standardization of numerical features
   - one-hot encoding of categorical features
4. **Train-test split**
5. **Model training with scikit-learn**
6. **Model evaluation**
7. **Export of the trained model to JSON**
8. **Preparation of a static web interface**

## Model
The model used in this phase is:

- **Algorithm:** Linear Regression
- **Library:** scikit-learn

A linear model was selected because it is:
- simple and interpretable,
- fast to train,
- easy to export to JSON,
- suitable for browser-based inference.

## Evaluation Metrics
The model is evaluated using:

- **MAE (Mean Absolute Error)**
- **RMSE (Root Mean Squared Error)**
- **R² Score**

The notebook also generates:
- a **residual distribution plot**
- a **predicted vs actual plot**


## Author
Ben Schmidtberger
Jonas Lehmann
Moritz Poettschacher

