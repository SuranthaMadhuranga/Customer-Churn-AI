import streamlit as st
from textwrap import dedent


from utils import (
    load_css,
    get_customer_summary,
    predict_customer,
    get_prediction_theme,
    get_shap_explanation,
    get_top_shap_features,
)

# =====================================================
# Prediction Page
# =====================================================


def show_prediction(df):

    # =====================================================
    # Load CSS
    # =====================================================

    load_css("app/styles/prediction.css")

    # =====================================================
    # Page Title
    # =====================================================

    st.markdown(
        """
        <div class="dashboard-title">
        🔮 Customer Churn Prediction
        </div>
        """,
        unsafe_allow_html=True,
    )

    # =====================================================
    # Search Customer
    # =====================================================

    st.markdown(
        """
        <div class="dashboard-subtitle">
        🔍 Search Customer
        </div>
        """,
        unsafe_allow_html=True,
    )

    search_col, button_col = st.columns([3, 1])

    with search_col:

        customer_id = st.text_input(
            label="Customer ID",
            placeholder="Enter Customer ID...",
            label_visibility="collapsed",
        )

    with button_col:

        search = st.button(
            "🔍 Search",
            use_container_width=True,
        )

    # =====================================================
    # Divider
    # =====================================================

    st.markdown(
        '<div class="dashboard-divider"></div>',
        unsafe_allow_html=True,
    )

    # =====================================================
    # Wait Until Search
    # =====================================================

    if not search:
        return

    # =====================================================
    # Validation
    # =====================================================

    customer_id = customer_id.strip()

    if customer_id == "":

        st.warning("Please enter a Customer ID.")

        return

    # =====================================================
    # Search Customer
    # =====================================================

    customer = df[df["customerID"] == customer_id]

    if customer.empty:

        st.error("Customer ID not found.")

        return

    customer = customer.iloc[0]

    # =====================================================
    # Customer Summary
    # =====================================================

    st.markdown(
        """
        <div class="dashboard-subtitle">
        📋 Customer Summary
        </div>
        """,
        unsafe_allow_html=True,
    )

    summary = get_customer_summary(customer)

    rows = []

    for label, value in summary.items():
        rows.append(
            f'<div class="summary-row-p">'
            f'<span class="summary-label-p">{label}</span>'
            f'<span class="summary-value-p">{value}</span>'
            f"</div>"
        )

    html = '<div class="summary-card-p">' + "".join(rows) + "</div>"

    st.markdown(html, unsafe_allow_html=True)

    # =====================================================
    # Prediction
    # =====================================================

    customer_df = customer.to_frame().T

    probability, prediction = predict_customer(customer_df)

    theme = get_prediction_theme(probability)

    percentage = probability * 100

    # =====================================================
    # Divider
    # =====================================================

    st.markdown(
        '<div class="dashboard-divider"></div>',
        unsafe_allow_html=True,
    )

    # =====================================================
    # Prediction Result
    # =====================================================

    st.markdown(
        """
        <div class="dashboard-subtitle">
        🎯 Prediction Result
        </div>
        """,
        unsafe_allow_html=True,
    )

    # =====================================================
    # Progress Bar
    # =====================================================

    progress_html = dedent(f"""
        <div class="progress-container">
        <div class="progress-track">
        <div
    class="progress-fill"
    style="
    width:{percentage:.2f}%;
    background:{theme['color']};
    ">
    </div>
    </div>
    </div>
    """)

    st.markdown(
        progress_html,
        unsafe_allow_html=True,
    )

    # =====================================================
    # Prediction Card
    # =====================================================

    prediction_html = dedent(f"""
        <div
    class="prediction-card"
    style="
    background:{theme['background']};
    border:1px solid {theme['border']};
    ">

    <div class="prediction-title">
    Churn Probability
    </div>

    <div
    class="prediction-percent"
    style="color:{theme['color']};">
    {percentage:.2f}%
    </div>

    <div
    class="prediction-type"
    style="color:{theme['color']};">
    {theme['type']}
    </div>

    </div>
    """)

    st.markdown(
        prediction_html,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.subheader("🧠 SHAP Explanation")
    fig = get_shap_explanation(customer_df)

    st.pyplot(fig, use_container_width=True)

    top_features = get_top_shap_features(customer_df, top_n=6)

    st.markdown("### 📌 Top 6 Feature Contributions")

    for _, row in top_features.iterrows():

        feature = row["Feature"]

        contribution = row["SHAP"]

        if contribution >= 0:
            icon = "🔺"
            color = "#EF4444"
            direction = "Increased Churn Risk"

        else:
            icon = "🔻"
            color = "#22C55E"
            direction = "Reduced Churn Risk"

        st.markdown(
            f"""
            <div class="shap-card">

            <div class="shap-header">

            <span>{icon} {feature}</span>

            <span style="color:{color};font-weight:700;">
            {contribution:.3f}
            </span>

            </div>

            <div class="shap-footer">

            {direction}

            </div>

            </div>
            """,
            unsafe_allow_html=True,
        )
