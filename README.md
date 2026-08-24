# Python-Data-Project
# Customer Churn Prediction and Retention Analytics Using Machine Learning 

## 1.Project Overview 
Customer churn refers to customers discontinuing their relationship with a company or services.
Predicting customer churn helps businesses identify customers who are likely to leave and take proactive steps to improve customer retention.
This project develops a Machin Learning-based system to predict customer churn and analyze the factors that influence customer attrition.

## 2.Problem Statement 
High Customer Churn can lead to significant revenue loss for businesses.
Therefore,identifying customers who are likely to churn in advance is important.
This project aims to develope a Machine Learning Model that can predict customer churn using customer demographic,contractual and service-releated information.

## 3.Objectives
- Develop an automated Machine Learning pipeline for predicting customer churn.
- Analyze important factors influencing customer attrition.
- Preprocess and transform customer data for machine Learning.
- Train a Random Forest Classification model.
- Evalute the model using Accuracy, Precision, Recall and F1-Score.
- visualize important features affecting customer churn.

## 4.Scope of the Project
The Project focuses on binary classification of customers into:

- 0 → Customer does not churn
- 1 → Customer churns

The approach can be extended to subscription-based businesses such as telecommunication, banking, insurance, streaming and other service industries.

## 5.Literature Review 
Machine Learning techniques arre widely used for customer analytics and churn prediction.

Commonly used approaches include:
-Logistic Regression
-Decision Trees
-Random Forest
-Ensemble Learning

Traditional statical approaches can provide useful relationships between variables, while Machine Learning algorithms can capture more complex and non-linear relationships within customer data.
In this project, Random Forest Classifier is selected because it can handle multiple features and provide features importance information.

## 6.System Architecture
The overall workflow of the project is:

Data Collection
↓
Data Preprocessing
↓
Feature Encoding
↓
Train-Test Split
↓
Feature Scaling
↓
Model Training
↓
Prediction
↓
Model Evaluation
↓
Feature Importance Analysis

## 7.Dataset Description
A synthetic dataset containing 1000 customer records is generated for this project.

Feature:

| CustomerID | Unique customer identification number |
| Age | Age of the customer |
| Tenure_Months | Duration of customer relationship in months |
| MonthlyCharges | Monthly amount paid by the customer |
| ContractType | Type of customer contract |
| TechSupport | Whether technical support is available |
| Churn | Target variable indicating customer churn |

## 8.Data preprocessing

The following preprocessing steps are performed:

1. CustomerID is removed because it is an identifier and does not provide predictive    information.
2. Categorical variables are converted into numerical values using Label Encoding.
3. Features and target variables are separated.
4. Dataset is divided into training and testing sets using an 80:20 ratio.
5. StandardScaler is used for feature scaling.

## 9.Algorithm used

Random Forest Classifier

Random Forest is an ensemble Machine Learning algorithm that combines multiple decision 
trees to perform classification.
In this project, Random Forest is used to predict whether a customer will churn.
The model is also used to determine the relative importance of different customer features.

## 10.Implementation

Technologies Used:
- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

Main Libraries:
python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

Machine Learning Component:
* train_test_split
* StandardScaler
* LabelEncoder
* RandomForestClassifier
* accuracy_score
* classification_report
* confusion_matrix

## 11.Add Result

--- Model Performance ---
Accuracy: 75.00%
Classification Report:
    Classification Report:
              precision    recall  f1-score   support
           0       0.76      0.97      0.85       151
           1       0.43      0.06      0.11        49
    accuracy                           0.75       200
   macro avg       0.60      0.52      0.48       200
weighted avg       0.68      0.75      0.67       200


### Feature Importance

The Random Forest model is also used to identify the most important features influencing 
customer churn.
The feature importance graph helps understand which customer characteristics contribute 
most to the prediction.
![image alt](https://github.com/Mohini-Suryawanshi-27/Python-Data-Project/blob/552ed618504046ab022d000f69e72e46f3e60305/Figure_1.png)
## 12.Future Scope

The project can be further improved by:

- Using a real-world customer churn dataset.
- Comparing Random Forest with Logistic Regression, Decision Tree, and other ML algorithms.
- Performing hyperparameter tuning.
- Developing a web application using Streamlit or Flask.
- Deploying the model online.
- Integrating real-time customer data.
- Implementing advanced Machine Learning techniques.

## 13.conclusion

This project demonstrates the use of Machine Learning for customer churn prediction.

A complete Machine Learning pipeline was developed, including data generation, 
preprocessing, feature encoding, model training, prediction, evaluation, and feature 
importance analysis.

The Random Forest Classifier provides a practical approach for identifying customers 
who may be at risk of churn and can help businesses develop effective customer retention 
strategies.

## 14.References

1. Scikit-learn Documentation
2. Python Documentation
3. Pandas Documentation
4. NumPy Documentation
5. Matplotlib Documentation
6. Research papers and academic literature related to customer churn prediction.




