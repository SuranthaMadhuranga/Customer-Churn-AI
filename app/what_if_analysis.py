import streamlit as st
from streamlit_extras.stylable_container import stylable_container
from utils import (
    load_css,
    get_customer,
    get_customer_summary,
    simulate_customer,
    predict_customer,
    get_risk_level,
)


def show_what_if_analysis(df):

    # =====================================================
    # Load CSS
    # =====================================================

    load_css("app/styles/main.css")
    load_css("app/styles/what_if_analysis.css")

    # =====================================================
    # Page Title
    # =====================================================

    st.markdown(
        """
        <div class="dashboard-title">
        🔄 What-If Analysis
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

    col1, col2 = st.columns([3, 1])

    with col1:

        customer_id = st.text_input(
            "Customer ID",
            placeholder="Enter Customer ID",
            key="whatif_customer_id",
            label_visibility="collapsed",
        )

    with col2:

        search = st.button(
            "🔍 Search",
            use_container_width=True,
            key="whatif_search_button",
        )

    # =====================================================
    # Search Result
    # =====================================================

    if search:

        customer = get_customer(df, customer_id)

        if customer is None:

            st.error("Customer not found.")

            return

        # Save customer for future reruns
        st.session_state["customer"] = customer

    # =====================================================
    # Load Selected Customer
    # =====================================================

    if "customer" not in st.session_state:
        return

    customer = st.session_state["customer"]
    summary = get_customer_summary(customer)

    # =====================================================
    # Customer Summary
    # =====================================================

    st.markdown(
        """
    <div class="whatif-section-title">
    📋 Customer Summary
    </div>
    """,
        unsafe_allow_html=True,
    )

    rows = []

    for label, value in summary.items():

        rows.append(
            f'<div class="summary-row">'
            f'<span class="summary-label">{label}</span>'
            f'<span class="summary-value">{value}</span>'
            f"</div>"
        )

    summary_html = '<div class="summary-card">' + "".join(rows) + "</div>"

    st.markdown(summary_html, unsafe_allow_html=True)

    # =====================================================
    # Customer Settings
    # =====================================================

    st.divider()

    st.markdown(
        """
        <h2 style='text-align:center; margin-bottom:15px;'>
        ⚙️ Customer Settings
        </h2>
        """,
        unsafe_allow_html=True,
    )

    # =====================================================
    # Customer Settings
    # =====================================================

    left_space, left, right, right_space = st.columns([0.15, 1, 1, 0.15])

    # =====================================================
    # Left Column
    # =====================================================

    with left:

        contract = st.selectbox(
            "Contract",
            [
                "Month-to-month",
                "One year",
                "Two year",
            ],
            index=[
                "Month-to-month",
                "One year",
                "Two year",
            ].index(customer["Contract"]),
            key="whatif_contract",
        )

        tenure = st.number_input(
            "Tenure",
            min_value=0,
            max_value=100,
            value=int(customer["tenure"]),
            key="whatif_tenure",
        )

        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            value=float(customer["MonthlyCharges"]),
            key="whatif_monthly_charges",
        )

        internet_service = st.selectbox(
            "Internet Service",
            [
                "DSL",
                "Fiber optic",
                "No",
            ],
            index=[
                "DSL",
                "Fiber optic",
                "No",
            ].index(customer["InternetService"]),
            key="whatif_internet_service",
        )

    # =====================================================
    # Right Column
    # =====================================================

    with right:

        online_security = st.selectbox(
            "Online Security",
            [
                "Yes",
                "No",
                "No internet service",
            ],
            index=[
                "Yes",
                "No",
                "No internet service",
            ].index(customer["OnlineSecurity"]),
            key="whatif_online_security",
        )

        online_backup = st.selectbox(
            "Online Backup",
            [
                "Yes",
                "No",
                "No internet service",
            ],
            index=[
                "Yes",
                "No",
                "No internet service",
            ].index(customer["OnlineBackup"]),
            key="whatif_online_backup",
        )

        tech_support = st.selectbox(
            "Tech Support",
            [
                "Yes",
                "No",
                "No internet service",
            ],
            index=[
                "Yes",
                "No",
                "No internet service",
            ].index(customer["TechSupport"]),
            key="whatif_tech_support",
        )

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
            index=[
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ].index(customer["PaymentMethod"]),
            key="whatif_payment_method",
        )

    # =====================================================
    # Edited Customer Data
    # =====================================================

    edited_customer = {
        "Contract": contract,
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "TechSupport": tech_support,
        "PaymentMethod": payment_method,
    }

    left_btn, center_btn, right_btn = st.columns([1, 2, 1])

    with center_btn:

        simulate = st.button(
            "🚀 Simulate Changes",
            use_container_width=True,
            key="whatif_simulate",
        )

    if simulate:

        # ===============================================
        # Original Customer Prediction
        # ===============================================

        original_customer = customer.to_frame().T

        original_probability, _ = predict_customer(original_customer)

        original_risk = get_risk_level(original_probability)

        # ===============================================
        # Simulated Customer Prediction
        # ===============================================

        simulated_customer = simulate_customer(
            customer,
            edited_customer,
        )

        simulated_probability, _ = predict_customer(simulated_customer)

        simulated_risk = get_risk_level(simulated_probability)

        # ===============================================
        # Risk Change
        # ===============================================

        risk_change = (simulated_probability - original_probability) * 100

        # ===============================================
        # Simulation Results
        # ===============================================

        st.divider()

        st.markdown(
            """
            <div class="whatif-section-title">
            📊 Simulation Results
            </div>
            """,
            unsafe_allow_html=True,
        )

        result_left, result_right = st.columns(2)

        # ---------------------------------------
        # Original Customer
        # ---------------------------------------

        with result_left:

            risk_class = ""

            if "Low" in original_risk:
                risk_class = "low-risk"

            elif "Medium" in original_risk:
                risk_class = "medium-risk"

            elif "At Risk" in original_risk:
                risk_class = "at-risk"

            else:
                risk_class = "high-risk"

            simulated_html = f"""
            <div class="result-card">

            <div class="result-title">
            🔄 Simulated Customer
            </div>

            <div class="result-probability">
            {simulated_probability * 100:.2f}%
            </div>

            <div class="result-risk {risk_class}">
            {simulated_risk}
            </div>

            </div>
            """

            st.markdown(
                simulated_html,
                unsafe_allow_html=True,
            )

        # ---------------------------------------
        # Simulated Customer
        # ---------------------------------------

        with result_right:

            risk_class = ""

            if "Low" in simulated_risk:
                risk_class = "low-risk"

            elif "Medium" in simulated_risk:
                risk_class = "medium-risk"

            elif "At Risk" in simulated_risk:
                risk_class = "at-risk"

            else:
                risk_class = "high-risk"

            original_html = f"""
            <div class="result-card">

            <div class="result-title">
            👤 Original Customer
            </div>

            <div class="result-probability">
            {original_probability * 100:.2f}%
            </div>

            <div class="result-risk {risk_class}">
            {original_risk}
            </div>

            </div>
            """

            st.markdown(
                original_html,
                unsafe_allow_html=True,
            )
        # ===============================================
        # Risk Change
        # ===============================================

        st.divider()

        if risk_change > 0:

            st.markdown(
                f"""
            <div class="risk-change risk-up">
            📈 Risk Increased by {risk_change:.2f}%
            </div>
            """,
                unsafe_allow_html=True,
            )

        elif risk_change < 0:

            st.markdown(
                f"""
            <div class="risk-change risk-down">
            📉 Risk Decreased by {abs(risk_change):.2f}%
            </div>
            """,
                unsafe_allow_html=True,
            )

        else:

            st.markdown(
                """
            <div class="risk-change risk-same">
            ➖ No Risk Change
            </div>
            """,
                unsafe_allow_html=True,
            )
