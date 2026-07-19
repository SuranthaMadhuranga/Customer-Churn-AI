import streamlit as st
import plotly.express as px
from utils import (
    predict_dataset,
    load_css,
    render_customer_table,
)


def show_customer_analytics(df):

    # =====================================================
    # Load CSS
    # =====================================================

    load_css("app/styles/customer_analytics.css")

    # =====================================================
    # Predict Dataset
    # =====================================================

    df = predict_dataset(df)

    # =====================================================
    # Customer Status
    # =====================================================

    def customer_status(prob):

        if prob >= 55:
            return "At Risk"

        elif prob >= 15:
            return "Regular"

        elif prob >= 5:
            return "Loyal"

        else:
            return "VIP"

    df["Customer_Status"] = df["Churn_Probability"].apply(customer_status)

    # =====================================================
    # Page Title
    # =====================================================

    st.markdown(
        """
        <div class="dashboard-title">
        👥 Customer Analytics
        </div>
        """,
        unsafe_allow_html=True,
    )

    # =====================================================
    # Customer Health Overview
    # =====================================================

    st.markdown(
        """
        <div class="dashboard-subtitle">
        📊 Customer Health Overview
        </div>
        """,
        unsafe_allow_html=True,
    )

    vip = (df["Customer_Status"] == "VIP").sum()
    loyal = (df["Customer_Status"] == "Loyal").sum()
    regular = (df["Customer_Status"] == "Regular").sum()
    at_risk = (df["Customer_Status"] == "At Risk").sum()

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.markdown(
            f"""
            <div class="metric-card-customer metric-vip">
            <div class="metric-label">👑 VIP</div>
            <div class="metric-value">{vip:,}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:

        st.markdown(
            f"""
            <div class="metric-card-customer metric-loyal">
            <div class="metric-label">💚 LOYAL</div>
            <div class="metric-value">{loyal:,}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:

        st.markdown(
            f"""
            <div class="metric-card-customer metric-regular">
            <div class="metric-label">🔵 REGULAR</div>
            <div class="metric-value">{regular:,}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:

        st.markdown(
            f"""
            <div class="metric-card-customer metric-atrisk">
            <div class="metric-label">🔴 AT RISK</div>
            <div class="metric-value">{at_risk:,}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="dashboard-divider"></div>',
        unsafe_allow_html=True,
    )

    # =====================================================
    # Customer Segment Distribution
    # =====================================================

    st.markdown(
        """
        <div class="dashboard-subtitle">
        📊 Customer Segment Distribution
        </div>
        """,
        unsafe_allow_html=True,
    )

    segment_df = (
        df["Customer_Status"]
        .value_counts()
        .rename_axis("Customer_Status")
        .reset_index(name="Customers")
    )

    order = [
        "At Risk",
        "Regular",
        "Loyal",
        "VIP",
    ]

    colors = {
        "VIP": "#FACC15",
        "Loyal": "#22C55E",
        "Regular": "#3B82F6",
        "At Risk": "#EF4444",
    }

    segment_fig = px.bar(
        segment_df,
        x="Customers",
        y="Customer_Status",
        orientation="h",
        color="Customer_Status",
        text="Customers",
        category_orders={"Customer_Status": order},
        color_discrete_map=colors,
    )

    segment_fig.update_traces(
        textposition="outside",
    )

    segment_fig.update_layout(
        template="plotly_dark",
        height=430,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        margin=dict(
            l=20,
            r=20,
            t=20,
            b=20,
        ),
        font=dict(
            color="white",
            size=15,
        ),
        xaxis=dict(
            title="<b>Customers</b>",
            showgrid=True,
            gridcolor="rgba(255,255,255,.08)",
            title_font=dict(size=15),
            tickfont=dict(size=13),
        ),
        yaxis=dict(
            title="",
            tickfont=dict(size=15),
        ),
    )

    st.plotly_chart(
        segment_fig,
        use_container_width=True,
        config={"displayModeBar": False},
    )

    st.markdown(
        '<div class="dashboard-divider"></div>',
        unsafe_allow_html=True,
    )

    # =====================================================
    # Top 10 At Risk Customers
    # =====================================================

    st.markdown(
        """
        <div class="dashboard-subtitle">
        🔥 Top 10 At Risk Customers
        </div>
        """,
        unsafe_allow_html=True,
    )

    risk_df = (
        df[df["Customer_Status"] == "At Risk"]
        .sort_values("Churn_Probability", ascending=False)
        .head(10)[
            [
                "customerID",
                "Contract",
                "tenure",
                "MonthlyCharges",
                "Customer_Status",
            ]
        ]
    )

    risk_df.columns = [
        "Customer ID",
        "Contract",
        "Tenure (Months)",
        "Monthly Charges ($)",
        "Customer Status",
    ]

    risk_df["Monthly Charges ($)"] = risk_df["Monthly Charges ($)"].map(
        lambda x: f"${x:,.2f}"
    )

    risk_df = risk_df.reset_index(drop=True)

    render_customer_table(risk_df)

    st.markdown(
        '<div class="dashboard-divider"></div>',
        unsafe_allow_html=True,
    )

    # =====================================================
    # Top 10 VIP Customers
    # =====================================================

    st.markdown(
        """
        <div class="dashboard-subtitle">
        👑 Top 10 VIP Customers
        </div>
        """,
        unsafe_allow_html=True,
    )

    vip_df = (
        df[df["Customer_Status"] == "VIP"]
        .sort_values("Churn_Probability", ascending=True)
        .head(10)[
            [
                "customerID",
                "Contract",
                "tenure",
                "MonthlyCharges",
                "Customer_Status",
            ]
        ]
    )

    vip_df.columns = [
        "Customer ID",
        "Contract",
        "Tenure (Months)",
        "Monthly Charges ($)",
        "Customer Status",
    ]

    vip_df["Monthly Charges ($)"] = vip_df["Monthly Charges ($)"].map(
        lambda x: f"${x:,.2f}"
    )

    vip_df = vip_df.reset_index(drop=True)

    render_customer_table(vip_df)
