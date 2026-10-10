# Customer Churn Prediction System

An AI-powered customer analytics platform that combines Machine Learning and Explainable AI to predict customer churn, understand the reasons behind customer decisions, and evaluate retention strategies through interactive business intelligence.

## Overview

Customer churn is a major challenge for businesses, as losing existing customers can negatively affect revenue, profitability, and long-term growth. Identifying customers who are likely to leave before they actually do enables businesses to take proactive retention measures.

The **Customer Churn Prediction System** is an interactive Machine Learning application built with Python and Streamlit. Developed using the IBM Telco Customer Churn dataset containing 7,043 customer records, the system predicts churn probabilities, identifies at-risk customers, explains individual predictions using SHAP, and simulates potential retention strategies.

By integrating predictive analytics, customer segmentation, and Explainable AI, the platform transforms customer data into actionable insights that support data-driven retention decisions.

## Key Features

- **Churn Prediction Dashboard** — Monitor customer statistics, predicted churners, active customers, and overall churn risk.
- **Machine Learning Prediction** — Predict customer churn probabilities using a tuned LightGBM model.
- **Customer Risk Segmentation** — Categorize customers into VIP, Loyal, Regular, and At Risk segments.
- **Customer Analytics** — Identify the customers most likely to leave and those with the lowest predicted churn risk.
- **Individual Customer Prediction** — Search for a customer and explore their churn probability, risk category, and profile.
- **Explainable AI with SHAP** — Understand the factors influencing individual predictions through SHAP waterfall plots and feature contributions.
- **What-If Analysis** — Simulate changes to customer contracts, charges, subscribed services, and payment methods to compare potential churn risks.
- **Interactive Visualizations** — Explore customer behavior and model predictions using interactive Plotly charts.
- **Business-Focused Decision Threshold** — Apply a 0.40 probability threshold to identify potential churners and support retention prioritization.

## Machine Learning and AI Modules

### 1. Exploratory Data Analysis (EDA)

Analyzes customer information to understand churn patterns and identify factors associated with customer retention.

**Capabilities:**

- Examine churn distributions across customer demographics, subscribed services, and billing information.
- Investigate relationships between churn, contract type, customer tenure, and monthly charges.
- Identify important churn-related patterns, including the relationship between contract type and churn (-0.397 correlation) and tenure and churn (-0.352 correlation).

**Business value:** Helps businesses understand customer behavior and identify potential retention opportunities before developing predictive models.

### 2. Data Preprocessing and Feature Engineering

Prepares customer records for Machine Learning by cleaning the data and transforming it into a suitable format.

**Capabilities:**

- Clean customer records and handle missing or blank values in `TotalCharges`.
- Remove customer identifiers from model training.
- Convert categorical variables into numerical representations using One-Hot Encoding.
- Scale numerical features using StandardScaler.
- Identify influential features using Mutual Information and Random Forest feature importance.
- Save preprocessing objects to maintain consistency between training and application predictions.

**Business value:** Improves data quality, supports reliable model training, and ensures new customer records are processed consistently.

### 3. Customer Churn Prediction — LightGBM

Trains and evaluates multiple gradient boosting algorithms to identify customers who may discontinue a service.

**Models evaluated:**

- XGBoost
- CatBoost
- LightGBM

**Capabilities:**

- Tune model hyperparameters using RandomizedSearchCV.
- Optimize model selection using ROC-AUC.
- Address class imbalance through model-specific weighting strategies.
- Evaluate performance using accuracy, precision, recall, F1 score, and ROC-AUC.
- Apply a custom decision threshold of 0.40 to the deployed LightGBM model.

**Model Evaluation Results**

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| XGBoost | 75.5% | 52.6% | 79.7% | 63.3% | 0.847 |
| CatBoost | 74.3% | 51.0% | 80.2% | 62.4% | 0.847 |
| **LightGBM (Deployed)** | **78.8%** | **59.6%** | **62.3%** | **60.9%** | **0.845** |

The evaluation was conducted on a test set containing 1,409 customers. LightGBM was selected for deployment because it achieved the highest accuracy and precision among the evaluated models.

**Business value:** Helps businesses prioritize customers who may leave and allocate retention resources more effectively.

### 4. Customer Risk Segmentation

Classifies customers into risk categories using their predicted churn probabilities.

**Customer groups include:**

- **VIP Customers:** Predicted churn probability below 5%.
- **Loyal Customers:** Predicted churn probability from 5% to below 15%.
- **Regular Customers:** Predicted churn probability from 15% to below 55%.
- **At-Risk Customers:** Predicted churn probability of 55% or higher.

**Capabilities:**

- Identify customers who require immediate retention attention.
- Highlight customers with relatively low predicted churn risk.
- Display the ten customers with the highest predicted churn risk.
- Display the ten customers with the lowest predicted churn risk.

**Business value:** Enables targeted customer retention strategies and helps businesses focus on the customers who need the most attention.

### 5. Explainable AI — SHAP

Uses SHAP (SHapley Additive exPlanations) to explain the predictions generated by the deployed LightGBM model.

**Capabilities:**

- Generate explanations for individual customer predictions.
- Visualize feature contributions using SHAP waterfall plots.
- Identify the six most influential features for an individual prediction.
- Show which features increase or decrease the model's predicted churn risk.

**Business value:** Makes Machine Learning predictions easier to interpret, helping business users understand why a customer has been classified as potentially at risk.

### 6. What-If Analysis — Retention Strategy Simulation

Allows users to explore how changes to a customer's profile may affect the predicted likelihood of churn.

**Capabilities:**

- Select a customer and load their existing profile.
- Modify contract type, monthly charges, internet service, online security, online backup, tech support, and payment method.
- Recalculate the predicted churn probability using the deployed model.
- Compare original and simulated churn probabilities and risk categories.

**Business value:** Helps businesses evaluate possible retention actions before implementing them. For example, a retention team can explore whether changing a customer's contract or adding technical support changes the model's predicted churn risk.

*Note: What-If results represent model-based simulations, not guaranteed outcomes or proof that a particular intervention will prevent churn.*

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, LightGBM, XGBoost, CatBoost |
| Explainable AI | SHAP |
| Data Visualization | Plotly, Matplotlib, Seaborn |
| Web Application | Streamlit, streamlit-extras, Custom CSS |
| Model Persistence | Joblib |
| Development Environment | Jupyter Notebook |
| Dataset | IBM Telco Customer Churn Dataset |

## Application Workflow

1. **Data Exploration** — Analyze the IBM Telco Customer Churn dataset to understand customer characteristics and churn patterns.
2. **Data Preparation** — Clean missing values, encode categorical variables, scale numerical features, and prepare model inputs.
3. **Feature Analysis** — Evaluate feature relevance using Mutual Information and Random Forest feature importance.
4. **Model Training** — Train, tune, and compare XGBoost, CatBoost, and LightGBM.
5. **Model Selection** — Select LightGBM for deployment and save the trained model and preprocessing components.
6. **Customer Analysis** — Upload a customer CSV file and explore churn statistics, risk segments, and individual predictions.
7. **Prediction Explanation** — Use SHAP to understand the factors contributing to each prediction.
8. **Retention Simulation** — Modify selected customer attributes and compare the resulting predicted churn risk.

## Project Objectives

- Develop a Machine Learning system to predict customer churn.
- Compare multiple gradient boosting algorithms and select a suitable deployment model.
- Identify customers with a high predicted probability of leaving.
- Explain individual model predictions using Explainable AI.
- Segment customers according to their predicted churn risk.
- Simulate potential retention strategies through What-If analysis.
- Present predictive insights through an interactive business analytics dashboard.
- Support proactive, data-driven customer retention decisions.

## Key Learning Outcomes

This project provided practical experience in:

- Exploratory Data Analysis and correlation analysis.
- Data cleaning, preprocessing, encoding, and feature scaling.
- Feature selection using Mutual Information and Random Forest importance.
- Training and hyperparameter tuning of XGBoost, CatBoost, and LightGBM.
- Handling class imbalance and selecting business-oriented decision thresholds.
- Evaluating classification models using accuracy, precision, recall, F1 score, and ROC-AUC.
- Interpreting Machine Learning predictions using SHAP.
- Developing interactive multi-page applications with Streamlit and Plotly.
- Saving and loading trained models and preprocessing objects using Joblib.
- Applying predictive analytics to practical customer retention problems.

## Future Improvements

Potential enhancements include:

- Combining multiple models through stacking or voting ensembles.
- Integrating customer lifetime value to estimate revenue at risk.
- Developing automated, model-based retention action recommendations.
- Connecting the application to live databases and customer relationship management systems.
- Introducing email or Slack alerts for high-value customers with elevated churn risk.
- Monitoring model performance and data drift over time.
- Exposing prediction capabilities through a REST API using FastAPI.
- Containerizing the application with Docker for easier deployment.
- Improving automated data validation and preprocessing for uploaded customer datasets.

## Conclusion

The Customer Churn Prediction System demonstrates how Machine Learning and Explainable AI can help businesses move beyond historical customer reporting toward proactive retention management.

By combining churn prediction, customer risk segmentation, SHAP-based explanations, and What-If simulation, the platform provides an integrated environment for understanding customer behavior and evaluating potential retention strategies.

The project highlights the practical value of combining predictive analytics with interpretable AI to transform customer data into meaningful business insights.

---

**Developed as a practical project exploring Machine Learning, Explainable AI, predictive analytics, and customer retention strategies.**
