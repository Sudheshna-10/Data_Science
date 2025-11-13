# ML Reinforcement: Heart Disease Prediction System  

### Overview  
This project demonstrates an end-to-end **Machine Learning workflow** for predicting the likelihood of heart disease using clinical and physiological parameters.  

It includes **data preprocessing**, **exploratory data analysis (EDA)**, **model training**, **evaluation**, and **deployment** using **Streamlit** for real-time predictions.  

The goal is to help in **early detection of heart disease** by analyzing patient data such as age, cholesterol, blood pressure, and chest pain type — providing a valuable tool for healthcare professionals.  
## Aim  
To develop a **machine learning model** that accurately predicts the presence of heart disease and deploy it as an interactive **Streamlit web application** for real-time use.
## Project Workflow  
1. **Data Understanding** – Load and inspect the `heart.csv` dataset.  
2. **Data Cleaning** – Handle duplicates, encode categorical variables, and scale numerical data.  
3. **Feature Engineering** – Analyze correlations and select the most impactful features.  
4. **Model Building** – Train multiple models including Logistic Regression, Decision Tree, SVM, Random Forest, KNN, and XGBoost.  
5. **Model Evaluation** – Compare performance metrics (Accuracy, Precision, Recall, F1-score).  
6. **Deployment** – Deploy the best-performing model (Random Forest) using **Streamlit**.

## Dataset Details  
- **File Used:** `heart.csv`  
- **Records:** 918  
- **Target Variable:** `HeartDisease` (1 = Disease, 0 = No Disease)

## Model Building (`train.py`)  
The `train.py` script handles data preprocessing, model training, and model saving.  

**Key Steps:**
- Encode categorical variables using `LabelEncoder`
- Train-Test split: 80% / 20%
- Model: **Random Forest Classifier**
- Evaluation Metric: **Accuracy Score (~87%)**
- Save model with `joblib`

## Deployment

The trained model was deployed using Streamlit Cloud for online interaction.

Users can access the app, input patient details, and receive real-time predictions instantly.
## Conclusion

This project successfully implements a complete machine learning pipeline for heart disease prediction — from raw data to deployment.

The deployed Streamlit app provides a simple yet powerful interface to assist healthcare professionals in early detection of heart disease, promoting data-driven medical insights.
