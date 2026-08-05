# Customer Churn Warning System

## 1. Project Overview

Customer retention is a critical challenge for e-commerce platforms. 
Identifying customers who are likely to stop purchasing enables companies to conduct targeted marketing campaigns and improve customer lifetime value.

This project develops an **e-commerce customer churn warning model** based on historical transaction data.

The main objectives are:

- Build customer-level behavioral features
- Predict customers with high churn risk
- Identify key factors influencing customer churn
- Generate actionable customer lists for coupon campaigns


The final output provides a ranked list of high-risk customers that can be directly used by marketing teams for customer retention strategies.


---

# 2. Business Problem

## Background

Traditional marketing campaigns often distribute coupons to a large number of customers, which may result in:

- High marketing costs
- Low conversion efficiency
- Poor customer targeting


## Solution

This project builds a machine learning pipeline:
Transaction Data
    ↓
Customer Behavior Analysis
    ↓
RFM Feature Engineering
    ↓
Logistic Regression Model
    ↓
Churn Probability Prediction
    ↓
High-risk Customer List
    ↓
Targeted Coupon Campaign



---

# 3. Dataset


## Dataset Information

Dataset:

**Online Retail Dataset**

Source:

UCI Machine Learning Repository


The dataset contains transaction records from a UK-based online retailer.


## Original Data

After data preprocessing:

- Transaction records: **397,924**
- Customer records: **4,339**


## Data Attributes


|Feature|Description|
|-|-|
|InvoiceNo|Transaction ID|
|StockCode|Product ID|
|Description|Product description|
|Quantity|Purchase quantity|
|InvoiceDate|Transaction date|
|UnitPrice|Product price|
|CustomerID|Customer identifier|
|Country|Customer location|



# 4. Data Processing


The raw transaction data is transformed into customer-level behavioral features.


Main preprocessing steps:

1. Remove transactions without customer identifiers
2. Remove invalid records with negative quantities
3. Calculate transaction value:
TotalPrice = Quantity × UnitPrice

4. Aggregate transaction records by customer



# 5. Customer Feature Engineering


## RFM Analysis


RFM analysis is used to represent customer purchasing behavior.


|Feature|Meaning|
|-|-|
|Recency|Days since last purchase|
|Frequency|Number of purchases|
|Monetary|Total spending amount|



Example:
Customer B:

Recency = 326 days

Frequency = 1 order

Monetary = 77183

→ High churn risk


---

# 6. Churn Definition


A customer is considered as churned when:

Recency > 90 days



This means the customer has not purchased within approximately three months.


## Churn Distribution


After labeling:

|Class|Number|
|-|-:|
|Active Customer|2890|
|Churn Customer|1449|


The dataset contains:

- 66.6% active customers
- 33.4% churn-risk customers


---

# 7. Machine Learning Model


## Logistic Regression


Logistic Regression is selected because:

- Suitable for binary classification
- Fast training speed
- High interpretability
- Allows analysis of feature contribution


Model input:


Recency

Frequency

Monetary



Model output:


Probability of Customer Churn



---

# 8. Model Evaluation


The model performance is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC


## Results



Accuracy: 0.99

AUC Score: 1.00



Classification performance:


|Class|Precision|Recall|F1-score|
|-|-|-|-|
|Active Customer|0.99|1.00|1.00|
|Churn Customer|1.00|0.98|0.99|


Confusion Matrix:



[[569, 0],
[5, 294]]



The model successfully identifies most high-risk customers while maintaining low false prediction rates.


---

# 9. Feature Interpretation


Logistic Regression coefficients are analyzed to understand customer churn factors.


|Feature|Coefficient|Interpretation|
|-|-:|-|
|Recency|11.443|Longer inactivity strongly increases churn probability|
|Frequency|0.223|Frequent purchases indicate stronger engagement|
|Monetary|-0.536|High-value customers are less likely to churn|


## Business Insights


- Customers who have not purchased recently should receive retention incentives.
- Frequent buyers show stronger loyalty.
- High-value customers should receive priority retention strategies.


---

# 10. High-risk Customer Identification


The model generates customer risk scores:

Output features:



CustomerID

Recency

Frequency

Monetary

Churn Probability



Example:


|CustomerID|Recency|Frequency|Monetary|Risk|
|-|-:|-:|-:|-:|
|17850|372|34|5391.21|1.0|
|13747|374|1|79.60|1.0|


These customers can be prioritized for:

- Coupon campaigns
- Customer recall activities
- Personalized marketing


---

# 11. Visualization


## Confusion Matrix


(Add image here)


images/confusion_matrix.png




## Feature Importance


(Add image here)


images/feature_importance.png

# 12. Technology Stack


## Programming Language

- Python


## Data Analysis

- Pandas
- NumPy


## Machine Learning

- Scikit-learn
- Logistic Regression


## Visualization

- Matplotlib
- Seaborn


---

# 13. Business Application


This project demonstrates a complete machine learning workflow:


Data Collection

↓

Feature Engineering

↓

Model Training

↓

Risk Prediction

↓

Business Decision Support



The generated customer risk list can help e-commerce companies improve customer retention efficiency through targeted marketing.


---

# 14. Future Improvements


Possible improvements:


- Compare Logistic Regression with XGBoost and Random Forest
- Apply SHAP for advanced model interpretation
- Build automated customer scoring pipeline
- Deploy prediction API using Flask/FastAPI
- Integrate with marketing platforms for automatic coupon delivery
