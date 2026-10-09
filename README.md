# 📊 Customer Churn Prediction System

An AI-powered analytics system that uses Machine Learning and Explainable AI to predict which telecom customers are likely to leave, explain why, and test retention strategies before acting on them.

## Overview

Losing a customer costs far more than keeping one, yet most businesses only find out a customer has churned after they are already gone. This project gives a retention team an early warning. It scores every customer with a churn probability, groups them by risk, and shows the factors behind each prediction.

The Customer Churn Prediction System is built with Python, LightGBM, SHAP, and Streamlit. It is trained on the IBM Telco Customer Churn dataset (7,043 customers). A user uploads a customer CSV file and gets an interactive dashboard with predictions, customer segments, per-customer explanations, and a What-If simulator for testing changes such as a longer contract or added tech support.

## Key Features

- **Churn Prediction**: Score every customer with a churn probability using a tuned LightGBM model.
- **Business-Tuned Decision Threshold**: Flag churners at a 0.40 probability threshold instead of the default 0.50, so fewer at-risk customers are missed.
- **Executive Dashboard**: View total customers, predicted churners, active customers, and churn rate, with interactive Plotly charts.
- **Customer Segmentation**: Group customers into **VIP**, **Loyal**, **Regular**, and **At Risk** by churn probability.
- **Top Customer Lists**: See the 10 customers most likely to churn and the 10 most loyal customers.
- **Individual Prediction**: Search any customer by ID and see their churn probability, risk level, and profile.
- **Explainable AI (SHAP)**: Show a SHAP waterfall plot and the top 6 features that pushed a prediction up or down.
- **What-If Analysis**: Change a customer's contract, charges, services, or payment method and compare the original and simulated churn risk side by side.

## Data Science and Machine Learning Modules

### 1. Exploratory Data Analysis (EDA)

Studies the structure of the dataset and the patterns behind churn.

Capabilities:

- Analyze the distribution of churn across demographics, services, and billing.
- Find relationships between churn and features such as contract type, tenure, and monthly charges.
- Identify the strongest drivers: **Contract** (correlation -0.397) and **tenure** (-0.352). Customers on longer contracts, and customers who have stayed longer, are much less likely to leave.

**Value:** Shows which customer behaviors matter before any model is built.

### 2. Data Preprocessing

Prepares the raw data for machine learning.

Capabilities:

- Clean the dataset and handle missing values in `TotalCharges`.
- Remove identifier columns such as `customerID` from training.
- Encode categorical variables with One-Hot Encoding.
- Scale numerical features with StandardScaler.
- Save the fitted encoder and scaler so the app processes new data exactly as the model was trained.

**Value:** Gives the models clean, consistent input and keeps training and deployment in sync.

### 3. Feature Engineering and Selection

Finds the features with the most influence on churn.

Capabilities:

- Rank features with Mutual Information scores.
- Measure feature importance with a Random Forest model.
- Compare statistical and model-based rankings to choose the final feature set.

**Value:** Focuses the models on the signals that actually predict churn.

### 4. Model Training and Comparison: XGBoost, CatBoost, LightGBM

Trains and compares three gradient boosting models.

Capabilities:

- Tune hyperparameters with `RandomizedSearchCV`, optimizing ROC-AUC.
- Handle class imbalance with `scale_pos_weight` (XGBoost) and balanced class weights (CatBoost).
- Evaluate each model with accuracy, precision, recall, F1 score, and ROC-AUC.
- Apply a custom 0.40 decision threshold to LightGBM.

Test set results (1,409 customers):

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| XGBoost | 0.755 | 0.526 | 0.797 | 0.633 | 0.847 |
| CatBoost | 0.743 | 0.510 | 0.802 | 0.624 | 0.847 |
| **LightGBM (deployed)** | **0.788** | **0.596** | 0.623 | 0.609 | 0.845 |

All three models reach a similar ROC-AUC of about 0.85. LightGBM is deployed because it has the highest accuracy and precision, so retention teams spend less effort on customers who were never going to leave.

**Value:** Picks the model that best balances catching churners against raising false alarms.

### 5. Explainable AI: SHAP

Explains why the model made each prediction.

Capabilities:

- Build a SHAP explainer for the deployed LightGBM model.
- Draw a waterfall plot for each customer's prediction.
- Rank the top 6 features that raised or lowered that customer's churn risk.

**Value:** Turns a black-box score into reasons a business user can understand and act on.

### 6. What-If Simulation

Tests retention strategies before they are applied.

Capabilities:

- Load any customer's current profile.
- Edit key features: contract, monthly charges, internet service, online security, online backup, tech support, and payment method.
- Re-run the model and compare the original and simulated churn probability and risk level.

**Value:** Shows which offer is most likely to keep a specific customer, such as moving them to a yearly contract or adding tech support.

## Customer Risk Levels

| Segment | Churn Probability | Meaning |
|---|---|---|
| 👑 VIP | Below 5% | Very loyal, almost no churn risk |
| 💚 Loyal | 5% to 15% | Stable customers |
| 🔵 Regular | 15% to 55% | Worth watching |
| 🔴 At Risk | 55% and above | Needs retention action now |

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, LightGBM, XGBoost, CatBoost |
| Explainable AI | SHAP |
| Visualization | Plotly, Matplotlib, Seaborn |
| Web Application | Streamlit, streamlit-extras, custom CSS |
| Model Storage | Joblib |
| Development | Jupyter Notebook |
| Dataset | IBM Telco Customer Churn (7,043 customers, 20 features) |

## Project Structure

```
Customer_Churn_AI/
├── app/
│   ├── churn_app.py            # Main Streamlit app (entry point)
│   ├── dashboard.py            # Dashboard tab
│   ├── customer_analytics.py   # Customer segmentation tab
│   ├── prediction.py           # Individual prediction + SHAP tab
│   ├── what_if_analysis.py     # What-If simulation tab
│   ├── utils.py                # Model loading, preprocessing, prediction helpers
│   └── styles/                 # CSS for each page
├── data/                       # Raw and cleaned Telco datasets
├── models/                     # Trained models, encoder, scaler, threshold, SHAP explainer
├── notebooks/                  # EDA, preprocessing, feature selection, models, SHAP
└── requirements.txt
```

## Application Workflow

1. **Explore**: Analyze the dataset and churn patterns in the EDA notebook.
2. **Prepare**: Clean, encode, and scale the data, then save the preprocessing objects.
3. **Train**: Tune and compare XGBoost, CatBoost, and LightGBM.
4. **Deploy**: Save the best model (LightGBM), its decision threshold, and a SHAP explainer.
5. **Upload**: Load a customer CSV file into the Streamlit app.
6. **Analyze**: Review churn statistics, customer segments, and individual predictions with explanations.
7. **Simulate**: Use the What-If tool to test retention strategies for a specific customer.

## Project Objectives

- Predict customer churn accurately with Machine Learning.
- Compare several gradient boosting algorithms and deploy the best one.
- Explain every prediction so business users can trust it.
- Segment customers by risk to focus retention efforts.
- Let users test retention strategies before applying them.
- Present the results through an easy-to-use web dashboard.

## Key Learning Outcomes

This project provided practical experience in:

- Exploratory data analysis and correlation analysis.
- Data cleaning, encoding, and feature scaling.
- Feature selection with Mutual Information and Random Forest importance.
- Training and tuning XGBoost, CatBoost, and LightGBM.
- Handling imbalanced classes and choosing a business-focused decision threshold.
- Evaluating classifiers with precision, recall, F1 score, and ROC-AUC.
- Explaining model predictions with SHAP.
- Building a multi-page interactive dashboard with Streamlit and Plotly.
- Saving and reusing a full ML pipeline with Joblib.

## Future Improvements

Potential enhancements include:

- Combine the three models in a stacking or voting ensemble.
- Add customer lifetime value to rank customers by revenue at risk.
- Recommend the best retention action for each customer automatically.
- Connect to a live database or CRM instead of CSV uploads.
- Send email or Slack alerts when a high-value customer becomes At Risk.
- Track model performance and data drift over time.
- Serve predictions through a REST API (FastAPI) and deploy to the cloud.
- Package the app with Docker.

## Conclusion

The Customer Churn Prediction System shows how Machine Learning and Explainable AI can turn customer data into clear, actionable insight. It goes beyond predicting who will leave: it explains why and lets the business test how to keep them.

By combining churn prediction, customer segmentation, SHAP explanations, and What-If simulation in one dashboard, the project helps businesses act early and keep more of their customers.

*Developed as a practical project exploring Machine Learning, Explainable AI, and customer analytics.*
