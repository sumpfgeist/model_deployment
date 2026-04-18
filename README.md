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

## Browser-Based Deployment
The trained model is exported to `docs/model.json`.

The web application is fully client-side and built with:
- **HTML**
- **CSS**
- **JavaScript**

The browser:
- loads `model.json`,
- reconstructs the preprocessing logic,
- computes predictions locally,
- displays the predicted insurance charges.

This means no backend server is required for inference.

## Running the Project Locally

### 1. Open the project folder
Make sure the project contains:
- `insurance.csv`
- `phase1.ipynb`
- the `docs/` folder

### 2. Run the notebook
Execute all cells in `phase1.ipynb`.

This generates:
- `docs/model.json`
- `docs/metrics.json`
- `docs/residuals.png`
- `docs/pred_vs_actual.png`

### 3. Start a local web server
From the `docs/` folder, run:

```bash
python -m http.server 8000
```

Then open:

```text
http://localhost:8000
```

## GitHub Pages Deployment
To publish the web app:

1. Create a new **public GitHub repository**
2. Upload:
   - `phase1.ipynb`
   - `insurance.csv`
   - the complete `docs/` folder
3. Open **Settings > Pages**
4. Under **Build and deployment**, select:
   - **Source:** Deploy from a branch
   - **Branch:** `main`
   - **Folder:** `/docs`
5. Save the settings
6. Wait for GitHub Pages to publish the site

### GitHub Pages URL
Add your public deployment link here after publishing:

```text
https://sumpfgeist.github.io/model_deployment/
```

## Files to Submit
For Phase 1, the submission should include:

- `phase1.ipynb`
- `insurance.csv`
- `docs/index.html`
- `docs/styles.css`
- `docs/predict.js`
- `docs/model.json`
- GitHub Pages URL in the documentation

## Notes
- The notebook and dataset should be stored in the same project directory.
- The `docs/` folder is used because it works well with GitHub Pages.
- The exported JSON contains both model parameters and preprocessing metadata so the browser predictions match the Python training pipeline.

## Author
Prepared for the MLOps project assignment, Phase 1.
