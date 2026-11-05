# B2B Invoice Late Payment Prediction & Collection Prioritization

## Project Overview

Late invoice payments can negatively affect a company's cash flow and financial planning. This project aims to predict the probability of late payments using historical B2B invoice data and machine learning techniques.

In addition to predicting late payments, the project provides a collection prioritization approach to help businesses identify high-risk invoices and improve their collection processes.

## Business Problem

Companies often manage large numbers of invoices with different payment dates, amounts, and customer characteristics.

Identifying which invoices are likely to be paid late can help collection teams allocate their resources more effectively.

The main objectives of this project are:

- Predict whether an invoice will be paid late.
- Estimate the probability of late payment.
- Identify factors associated with payment delays.
- Prioritize open invoices based on their estimated risk and financial value.

## Technologies Used

**Programming Language:** Python

**Data Analysis:** Pandas, NumPy

**Data Visualization:** Matplotlib, Seaborn

**Machine Learning:** Scikit-learn, CatBoost, XGBoost, LightGBM

**Model Evaluation:** Accuracy, Precision, Recall, F1-Score, ROC-AUC

**Other Tools:** Power BI, Streamlit

## Project Workflow

### 1. Data Preprocessing
- Data cleaning and duplicate removal
- Missing value analysis
- Date conversion
- Outlier handling
- Categorical variable encoding

### 2. Exploratory Data Analysis (EDA)
- Invoice amount distribution analysis
- Customer payment behavior analysis
- Late payment distribution analysis
- Relationships between invoice features and late payments

### 3. Feature Engineering
- Payment term duration
- Invoice posting and due months
- Historical customer late payment rates
- Customer payment delay statistics
- Previous invoice activity

### 4. Machine Learning
Several classification algorithms were compared, including:

- Logistic Regression
- K-Nearest Neighbors
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM
- CatBoost

CatBoost was selected for the final prediction model.

### 5. Collection Prioritization

For open invoices, the predicted probability of late payment is combined with the invoice amount.

**Priority Score = Late Payment Probability × Invoice Amount**

Invoices are ranked according to their priority scores and categorized into three levels:

- **HIGH:** High-priority invoices requiring closer attention.
- **MEDIUM:** Invoices suitable for regular follow-up.
- **LOW:** Lower-priority invoices requiring routine monitoring.

## Model Evaluation

The project uses Accuracy, Precision, Recall, F1-Score, and ROC-AUC to evaluate classification models.

Model comparison and hyperparameter optimization are included in the Python workflow.

## Business Value

This project demonstrates how machine learning can support accounts receivable management by:

- Identifying potentially late payments
- Supporting data-driven collection decisions
- Prioritizing invoices based on financial risk
- Improving the efficiency of collection activities

## Project Status

This project is under development. The current repository contains the Python analysis and modeling workflow.

Further improvements include reproducibility checks, validation of historical customer features, and additional documentation.

