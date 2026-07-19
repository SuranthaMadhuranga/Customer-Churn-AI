import streamlit as st
import pandas as pd

from utils import load_css

from dashboard import show_dashboard
from customer_analytics import show_customer_analytics
from prediction import show_prediction
from what_if_analysis import show_what_if_analysis

# =====================================================
# Page Configuration
# =====================================================

st.set_page_config(
    page_title="Customer Churn Prediction System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =====================================================
# Load Global CSS
# =====================================================

load_css("app/styles/main.css")


# =====================================================
# Main Title
# =====================================================

st.markdown(
    """
    <h1 class="main-title">
    📊 Customer Churn Prediction System
    </h1>
    """,
    unsafe_allow_html=True,
)

# =====================================================
# Upload Dataset
# =====================================================

with st.container():

    uploaded_file = st.file_uploader("Upload Customer Churn Dataset", type=["csv"])

# =====================================================
# Main Application
# =====================================================

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📊Dashboard",
            "👥Customer Analytics",
            "🔮Prediction",
            "🔄What-If Analysis",
        ]
    )

    with tab1:
        show_dashboard(df)

    with tab2:
        show_customer_analytics(df)

    with tab3:
        show_prediction(df)

    with tab4:
        show_what_if_analysis(df)

else:
    st.info("📁 Upload the cleaned Telco Customer Churn CSV file to begin.")
