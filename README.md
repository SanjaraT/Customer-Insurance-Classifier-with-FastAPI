--> Project Overview

This repository contains the complete machine learning training pipeline for a Customer Insurance Classification system. I performed data preprocessing and feature engineering, built a scikit-learn pipeline, trained a Random Forest classifier, and saved the final model (insurance_model.pkl) for deployment. The trained model predicts insurance risk categories based on customer demographic and lifestyle attributes and is later integrated into a separate FastAPI-based API repository for production use.

--> Model Details

Algorithm: Random Forest Classifier

Preprocessing: Encoding + Scaling pipeline

Input: Customer income, BMI, occupation, age group, lifestyle risk, city tier

Output: Insurance risk category

--> Related Repository

The deployed API using FastAPI is available here:

 https://github.com/SanjaraT/Insurance-Prediction-API