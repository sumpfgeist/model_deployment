import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st
from mlflow.tracking import MlflowClient

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Insurance Charges Predictor", layout="wide")

TRACKING_URI = "http://127.0.0.1:5000"
REGISTERED_MODEL_NAME = "BestRegressor_v1"
DATA_PATH = Path("insurance.csv")


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


def _get_latest_model_uri():
    client = MlflowClient(tracking_uri=TRACKING_URI)
    versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    if not versions:
        raise RuntimeError(
            f"No registered model named '{REGISTERED_MODEL_NAME}' was found in MLflow."
        )
    versions = sorted(versions, key=lambda v: int(v.version), reverse=True)
    return f"models:/{REGISTERED_MODEL_NAME}/{versions[0].version}"


@st.cache_resource
def load_registered_model():
    mlflow.set_tracking_uri(TRACKING_URI)
    try:
        return mlflow.sklearn.load_model(
            model_uri=f"models:/{REGISTERED_MODEL_NAME}/Production"
        )
    except Exception:
        fallback_uri = _get_latest_model_uri()
        return mlflow.sklearn.load_model(model_uri=fallback_uri)


@st.cache_resource
def fit_ols_for_confidence_intervals():
    df = load_data().copy()

    X = df.drop(columns=["charges"])
    y = df["charges"].astype(float)

    X_encoded = pd.get_dummies(X, drop_first=False)
    X_encoded = X_encoded.astype(float)

    X_const = sm.add_constant(X_encoded, has_constant="add")
    ols_model = sm.OLS(y, X_const).fit()

    return ols_model, list(X_encoded.columns)


def build_input_row(age, bmi, children, sex, smoker, region):
    return pd.DataFrame(
        {
            "age": [age],
            "sex": [sex],
            "bmi": [bmi],
            "children": [children],
            "smoker": [smoker],
            "region": [region],
        }
    )


def prepare_input_for_ols(input_df, feature_columns):
    encoded = pd.get_dummies(input_df, drop_first=False)
    encoded = encoded.astype(float)
    encoded = encoded.reindex(columns=feature_columns, fill_value=0.0)
    return encoded


def extract_feature_importance(model):
    if not hasattr(model, "named_steps"):
        return pd.DataFrame(columns=["feature", "coefficient", "abs_coefficient"])

    preprocess = model.named_steps.get("preprocess")
    reg = model.named_steps.get("model")

    if preprocess is None or reg is None:
        return pd.DataFrame(columns=["feature", "coefficient", "abs_coefficient"])

    if not hasattr(reg, "coef_"):
        return pd.DataFrame(columns=["feature", "coefficient", "abs_coefficient"])

    feature_names = list(preprocess.get_feature_names_out())
    coefs = np.asarray(reg.coef_)

    coef_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefs,
        }
    )
    coef_df["abs_coefficient"] = coef_df["coefficient"].abs()
    coef_df = coef_df.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)
    return coef_df


def plot_feature_importance(coef_df):
    top_df = coef_df.head(10).sort_values("coefficient")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top_df["feature"], top_df["coefficient"])
    ax.set_title("Top 10 Linear Model Coefficients")
    ax.set_xlabel("Coefficient value")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    return fig


df = load_data()
model = load_registered_model()
ols_model, ols_feature_columns = fit_ols_for_confidence_intervals()

st.title("Medical Insurance Charges Prediction Dashboard")
st.write(
    "This Streamlit application loads the registered MLflow model from Phase 2 and "
    "predicts medical insurance charges from user-provided feature values."
)

with st.sidebar:
    st.header("User Input")

    age = st.slider(
        "Age",
        min_value=int(df["age"].min()),
        max_value=int(df["age"].max()),
        value=int(df["age"].median()),
        step=1,
    )

    bmi = st.slider(
        "BMI",
        min_value=float(df["bmi"].min()),
        max_value=float(df["bmi"].max()),
        value=float(round(df["bmi"].median(), 1)),
        step=0.1,
    )

    children = st.slider(
        "Children",
        min_value=int(df["children"].min()),
        max_value=int(df["children"].max()),
        value=int(df["children"].median()),
        step=1,
    )

    sex = st.selectbox("Sex", options=sorted(df["sex"].unique().tolist()))
    smoker = st.selectbox("Smoker", options=sorted(df["smoker"].unique().tolist()))
    region = st.selectbox("Region", options=sorted(df["region"].unique().tolist()))

    predict_clicked = st.button("Predict Charges", use_container_width=True)

input_df = build_input_row(age, bmi, children, sex, smoker, region)

left_col, right_col = st.columns([1.1, 1.0])

with left_col:
    st.subheader("Selected Input")
    st.dataframe(input_df, use_container_width=True, hide_index=True)

    if predict_clicked:
        prediction = float(model.predict(input_df)[0])

        input_for_ols = prepare_input_for_ols(input_df, ols_feature_columns)
        pred_summary = ols_model.get_prediction(
            sm.add_constant(input_for_ols, has_constant="add")
        ).summary_frame(alpha=0.05)

        ci_low = float(pred_summary["mean_ci_lower"].iloc[0])
        ci_high = float(pred_summary["mean_ci_upper"].iloc[0])

        st.subheader("Prediction Output")
        st.metric("Predicted insurance charges", f"${prediction:,.2f}")
        st.success(
            f"Predicted price: ${prediction:,.2f} "
            f"[95% CI: ${ci_low:,.2f} - ${ci_high:,.2f}]"
        )

        st.caption(
            "The main prediction comes from the registered MLflow model. "
            "The confidence interval is estimated with a local OLS model fitted on the dataset."
        )
    else:
        st.info("Adjust the input values in the sidebar and click 'Predict Charges'.")

with right_col:
    st.subheader("Key Insights")
    coef_df = extract_feature_importance(model)

    if coef_df.empty:
        st.warning("Feature importance could not be extracted from the loaded model.")
    else:
        fig = plot_feature_importance(coef_df)
        st.pyplot(fig)
        st.dataframe(
            coef_df[["feature", "coefficient"]].head(10),
            use_container_width=True,
            hide_index=True,
        )

st.subheader("Model and System Information")
info_col1, info_col2, info_col3 = st.columns(3)

with info_col1:
    st.write("**MLflow Tracking URI**")
    st.code(TRACKING_URI)

with info_col2:
    st.write("**Registered Model**")
    st.code(REGISTERED_MODEL_NAME)

with info_col3:
    st.write("**Dataset Rows**")
    st.code(str(len(df)))
