import streamlit as st
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt
import pandas as pd


def render_customer_table(df):
    """
    Render a premium HTML table.
    """

    html = """
    <div class="executive-table">
     <table>
       <thead>
           <tr>
    """

    # -------------------------
    # Table Header
    # -------------------------

    for column in df.columns:
        html += f"<th>{column}</th>"

    html += """
            </tr>
        </thead>
        <tbody>
    """

    # -------------------------
    # Table Body
    # -------------------------

    for _, row in df.iterrows():

        html += "<tr>"

        for value in row:
            html += f"<td>{value}</td>"

        html += "</tr>"

    html += """
            </tbody>
        </table>
    </div>
    """

    st.markdown(html, unsafe_allow_html=True)


# =====================================================
# Load CSS
# =====================================================


def load_css(css_file):

    with open(css_file) as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True,
        )


# =====================================================
# Load Models
# =====================================================


@st.cache_resource
def load_models():

    model = joblib.load("models/lightgbm_model.pkl")

    encoder = joblib.load("models/encoder.pkl")

    scaler = joblib.load("models/scaler.pkl")

    feature_names = joblib.load("models/feature_names.pkl")

    threshold = joblib.load("models/threshold.pkl")

    SHAP_EXPLAINER = joblib.load("models/shap_explainer.pkl")

    return model, encoder, scaler, feature_names, threshold, SHAP_EXPLAINER


# =====================================================
# Training Feature Definition
# =====================================================

NUMERICAL_COLS = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

CATEGORICAL_COLS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]


# =====================================================
# Preprocess Uploaded Dataset
# =====================================================


def preprocess_data(df, encoder, scaler):

    X = df.copy()

    # -----------------------------------------
    # Remove columns not used by the model
    # -----------------------------------------

    for col in ["customerID", "Churn"]:

        if col in X.columns:
            X.drop(columns=col, inplace=True)

    # -----------------------------------------
    # Check required columns
    # -----------------------------------------

    required_columns = NUMERICAL_COLS + CATEGORICAL_COLS

    missing_columns = [col for col in required_columns if col not in X.columns]

    if missing_columns:

        raise ValueError(f"Missing required columns:\n{missing_columns}")

    # -----------------------------------------
    # Keep same order used during training
    # -----------------------------------------

    X = X[required_columns]

    # -----------------------------------------
    # Transform
    # -----------------------------------------

    X_num = scaler.transform(X[NUMERICAL_COLS])

    X_cat = encoder.transform(X[CATEGORICAL_COLS])

    X_final = np.hstack(
        (
            X_num,
            X_cat,
        )
    )

    return X_final


# =====================================================
# Predict Dataset
# =====================================================


def predict_dataset(df):

    model, encoder, scaler, feature_names, threshold, _ = load_models()

    X = preprocess_data(
        df,
        encoder,
        scaler,
    )

    # Probability of churn

    probabilities = model.predict_proba(X)[:, 1]

    # Apply deployment threshold

    predictions = (probabilities >= threshold).astype(int)

    result_df = df.copy()

    result_df["Predicted_Churn"] = np.where(
        predictions == 1,
        "Yes",
        "No",
    )

    result_df["Churn_Probability"] = (probabilities * 100).round(2)

    return result_df


# =====================================================
# Prediction Summary
# =====================================================


def get_prediction_summary(df):

    total_customers = len(df)

    churn_customers = (df["Predicted_Churn"] == "Yes").sum()

    retained_customers = (df["Predicted_Churn"] == "No").sum()

    churn_rate = (churn_customers / total_customers) * 100

    average_probability = df["Churn_Probability"].mean()

    return {
        "total_customers": total_customers,
        "churn_customers": churn_customers,
        "retained_customers": retained_customers,
        "churn_rate": churn_rate,
        "average_probability": average_probability,
    }


# =====================================================
# Predict Single Customer
# =====================================================


def predict_customer(customer_df):
    """
    Predict churn for a single customer.
    """

    model, encoder, scaler, _, threshold, _ = load_models()

    X = preprocess_data(
        customer_df,
        encoder,
        scaler,
    )

    probability = model.predict_proba(X)[0][1]

    prediction = "Yes" if probability >= threshold else "No"

    return probability, prediction


# =====================================================
# Prediction Theme
# =====================================================


def get_prediction_theme(probability):

    probability *= 100

    if probability < 5:

        return {
            "type": "🟢 VIP",
            "color": "#22C55E",
            "background": "#123524",
            "border": "#22C55E",
        }

    elif probability < 15:

        return {
            "type": "🟢 Loyal",
            "color": "#84CC16",
            "background": "#24381A",
            "border": "#84CC16",
        }

    elif probability < 55:

        return {
            "type": "🟠 Regular",
            "color": "#F59E0B",
            "background": "#4A3310",
            "border": "#F59E0B",
        }

    else:

        return {
            "type": "🔴 At Risk",
            "color": "#EF4444",
            "background": "#4A1D1D",
            "border": "#EF4444",
        }


# =====================================================
# Customer Summary
# =====================================================


def get_customer_summary(customer):

    return {
        "📄 Contract": customer["Contract"],
        "⏳ Tenure": f'{customer["tenure"]} Months',
        "💰 Monthly Charges": f'${customer["MonthlyCharges"]:.2f}',
        "🌐 Internet Service": customer["InternetService"],
        "💳 Total Charges": f'${customer["TotalCharges"]:.2f}',
        "🔒 Online Security": customer["OnlineSecurity"],
    }


def get_shap_explanation(customer_df):

    _, encoder, scaler, feature_names, _, shap_explainer = load_models()

    processed = preprocess_data(customer_df, encoder, scaler)

    processed = pd.DataFrame(processed, columns=feature_names)

    shap_values = shap_explainer(processed)

    plt.figure(figsize=(8, 6))

    shap.plots.waterfall(shap_values[0], show=False)

    fig = plt.gcf()

    plt.close()

    return fig


def get_top_shap_features(customer_df, top_n=6):

    _, encoder, scaler, feature_names, _, shap_explainer = load_models()

    processed = preprocess_data(customer_df, encoder, scaler)

    processed = pd.DataFrame(processed, columns=feature_names)

    shap_values = shap_explainer(processed)

    importance = pd.DataFrame({"Feature": feature_names, "SHAP": shap_values.values[0]})

    importance["Impact"] = importance["SHAP"].abs()

    importance = importance.sort_values("Impact", ascending=False)

    return importance.head(top_n)


# =====================================================
# Get Customer by Customer ID
# =====================================================


def get_customer(df, customer_id):
    """
    Search and return a customer by Customer ID.

    Returns
    -------
    pandas.Series
    Customer record if found.

    None
    If customer does not exist.
    """

    customer = df[df["customerID"] == customer_id]

    if customer.empty:
        return None

    return customer.iloc[0]


# =====================================================
# Simulate Customer
# =====================================================


def simulate_customer(customer, edited_customer):
    """
    Apply edited values to a customer record and
    return the updated customer.
    """

    simulated_customer = customer.copy()

    # Apply edited values
    for column, value in edited_customer.items():
        simulated_customer[column] = value

        # Recalculate Total Charges
        simulated_customer["TotalCharges"] = (
            simulated_customer["tenure"] * simulated_customer["MonthlyCharges"]
        )

    return simulated_customer.to_frame().T


# =====================================================
# Get Risk Level
# =====================================================


def get_risk_level(probability):
    """
    Convert churn probability into a customer risk level.

    Parameters
    ----------
    probability : float
    Churn probability (0-1)

    Returns
    -------
    str
    Risk level with emoji.
    """

    probability *= 100

    if probability < 20:
        return "🟢 Low Risk"

    elif probability < 40:
        return "🟡 Medium Risk"

    elif probability < 60:
        return "🟠 At Risk"

    else:
        return "🔴 High Risk"
