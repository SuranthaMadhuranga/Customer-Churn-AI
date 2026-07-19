import streamlit as st
import plotly.express as px
from utils import predict_dataset, load_css


def show_dashboard(df):

    load_css("app/styles/dashboard.css")

    df = predict_dataset(df)

    # =====================================================
    # Dashboard Title
    # =====================================================

    st.markdown(
        """
        <div class="dashboard-title">
        📊 Churn Dashboard
        </div>
        """,
        unsafe_allow_html=True,
    )

    # =====================================================
    # Executive Summary
    # =====================================================

    total_customers = len(df)
    churn_customers = (df["Predicted_Churn"] == "Yes").sum()
    active_customers = (df["Predicted_Churn"] == "No").sum()
    churn_rate = churn_customers / total_customers * 100

    st.markdown(
        """
        <div class="dashboard-subtitle">
        📈 Executive Summary
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f"""
            <div class="metric-card-dashboard">
            <div class="metric-label">👥 TOTAL CUSTOMERS</div>
            <div class="metric-value">{total_customers:,}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div class="metric-card-dashboard">
            <div class="metric-label">❌ PREDICTED CHURN</div>
            <div class="metric-value">{churn_customers:,}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f"""
            <div class="metric-card-dashboard">
            <div class="metric-label">✅ ACTIVE CUSTOMERS</div>
            <div class="metric-value">{active_customers:,}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:
        st.markdown(
            f"""
            <div class="metric-card-dashboard">
            <div class="metric-label">📉 CHURN RATE</div>
            <div class="metric-value">{churn_rate:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="dashboard-divider"></div>', unsafe_allow_html=True)

    # =====================================================
    # Average Churn Probability by Tenure
    # =====================================================

    st.markdown(
        """
        <div class="dashboard-subtitle">
        📈 Average Churn Probability by Tenure
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------
    # Average Churn Probability for each tenure
    # --------------------------------------------

    tenure_df = (
        df.groupby("tenure", as_index=False)["Churn_Probability"]
        .mean()
        .sort_values("tenure")
    )

    # Optional: Remove tenure = 0 (recommended)
    tenure_df = tenure_df[tenure_df["tenure"] > 0]

    # --------------------------------------------
    # Plot Line Chart
    # --------------------------------------------

    line_fig = px.line(
        tenure_df,
        x="tenure",
        y="Churn_Probability",
        markers=True,
    )

    # --------------------------------------------
    # Line Style
    # --------------------------------------------

    line_fig.update_traces(
        mode="lines+markers",
        line=dict(
            color="#7EC3FF",
            width=1.7,
        ),
        marker=dict(
            size=4,
            color="#7EC3FF",
            line=dict(
                width=1,
                color="#7EC3FF",
            ),
        ),
    )

    # --------------------------------------------
    # Layout
    # --------------------------------------------

    line_fig.update_layout(
        template="plotly_dark",
        height=600,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(
            l=20,
            r=120,
            t=20,
            b=20,
        ),
        font=dict(
            color="white",
            size=14,
        ),
        hovermode="x unified",
        xaxis=dict(
            title="<b>Tenure (Months)</b>",
            title_font=dict(
                size=15,
                color="white",
            ),
            showgrid=False,
            zeroline=False,
            showline=False,
        ),
        yaxis=dict(
            title="<b>Average Churn Probability (%)</b>",
            title_font=dict(
                size=15,
                color="white",
            ),
            ticksuffix="%",
            gridcolor="rgba(255,255,255,0.10)",
            zeroline=False,
            showline=False,
        ),
        showlegend=False,
    )

    # --------------------------------------------
    # plotly_chart
    # --------------------------------------------

    st.plotly_chart(
        line_fig,
        use_container_width=True,
        config={"displayModeBar": False},
    )

    st.markdown('<div class="dashboard-divider"></div>', unsafe_allow_html=True)

    # =====================================================
    # Customer Breakdown
    # =====================================================

    st.markdown(
        """
        <div class="dashboard-subtitle">
        📊 Customer Breakdown
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, space, col2 = st.columns([1, 0.0001, 1])

    churn_dist = df["Predicted_Churn"].value_counts().reset_index()

    churn_dist.columns = [
        "Predicted_Churn",
        "Customers",
    ]

    # ---------------- Donut ----------------

    with col1:

        donut = px.pie(
            churn_dist,
            values="Customers",
            names="Predicted_Churn",
            hole=0.60,
            color="Predicted_Churn",
            color_discrete_map={
                "Yes": "#EF4444",
                "No": "#22C55E",
            },
        )

        donut.update_layout(
            template="plotly_dark",
            height=470,
            title=dict(
                text="<b>Predicted Churn Distribution</b>",
                x=0.34,
                font=dict(
                    size=19,
                    color="white",
                ),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(
                orientation="h",
                x=0.5,
                xanchor="center",
                y=-0.08,
                font=dict(
                    size=13,  # Increase Yes/No font size
                    color="white",
                ),
                title=dict(
                    text="",  # Hide legend title (optional)
                ),
            ),
        )

        st.markdown('<div class="chart-card">', unsafe_allow_html=True)

        st.plotly_chart(
            donut,
            use_container_width=True,
            config={"displayModeBar": False},
        )

        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- Contract ----------------

    with col2:

        contract = px.histogram(
            df,
            x="Contract",
            color="Predicted_Churn",
            barmode="group",
            color_discrete_map={
                "Yes": "#EF4444",
                "No": "#22C55E",
            },
        )

        contract.update_layout(
            template="plotly_dark",
            height=470,
            title=dict(
                text="Contract vs Predicted Churn",
                x=0.34,
                font=dict(
                    size=19,
                    color="white",
                ),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(
                color="white",
            ),
            # X-axis
            xaxis=dict(
                title="Contract",
                title_font=dict(
                    size=16,
                ),
                tickfont=dict(
                    size=13,
                ),
            ),
            # Y-axis
            yaxis=dict(
                title="Customers",
                title_font=dict(
                    size=15,
                ),
                tickfont=dict(
                    size=13,
                ),
            ),
            # Legend
            legend=dict(
                x=1.02,  # Move to the right
                y=0.5,  # Middle vertically
                xanchor="left",
                yanchor="middle",
                font=dict(
                    size=15,
                    color="white",
                ),
                title=dict(
                    text="<b>Predicted Churn</b>",
                    font=dict(size=15),
                ),
            ),
        )

        st.markdown('<div class="chart-card">', unsafe_allow_html=True)

        st.plotly_chart(
            contract,
            use_container_width=True,
            config={"displayModeBar": False},
        )

        st.markdown("</div>", unsafe_allow_html=True)
