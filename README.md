# 🏃 Walk vs Run Activity Classification

This machine learning project focuses on building a predictive model to classify human activity — whether a person is **walking** or **running** — based on motion sensor data. The project is designed to showcase data preprocessing, feature analysis, model evaluation, and classification accuracy using real-world time-series features from motion sensors.

---

## 📌 Problem Statement

Classify user activity (walk or run) using motion sensor data (accelerometer and gyroscope) collected from wearable devices. The goal is to analyze how sensor variables influence classification and develop an accurate model that can be used for fitness apps, health tracking, or wearables.

---

## 📂 Dataset Overview

- 📈 Total Features: 6 (after preprocessing)
- 🧠 Target Variable: `activity` (Walk = 0, Run = 1)
- 📦 Source: Provided by DataMites (Subset of the original Run/Walk dataset)

| Column            | Description              |
|-------------------|--------------------------|
| acceleration_x/y/z | Accelerometer data (3D) |
| gyro_x/y/z         | Gyroscope data (3D)     |

---

## ⚙️ Technologies Used

- **Python**, **Scikit-learn**, **Pandas**, **Seaborn**, **Matplotlib**
- **GridSearchCV** for hyperparameter tuning
- **ML Models**: Logistic Regression, Random Forest, XGBoost, MLPClassifier
- **StandardScaler** for feature normalization
- **Train-Test Split** and **Stratified Sampling**

---

## 🔍 Exploratory Data Analysis (EDA)

- Analyzed feature distributions across walk/run activity using histograms & KDE plots
- Checked for outliers using boxplots and removed extreme low/misclassified values
- Correlation heatmap revealed strong signal in acceleration and gyro features



---

## 🧠 Models Compared

| Model              | Accuracy  | Notes                             |
|-------------------|-----------|-----------------------------------|
| Logistic Regression | 84%     | Underfitting, limited complexity  |
| Random Forest      | 99.3%     | Best performer, fast inference    |
| XGBoost            | 99.2%     | Similar to RF, slightly slower    |
| MLPClassifier      | 99%       | Tuned with GridSearchCV, stable   |

✅ **Best Model**: Random Forest (balanced accuracy, low overfitting)

---

## 📈 Evaluation Metrics

- Accuracy, Classification Report, Confusion Matrix
- 3-Fold Cross-Validation



---

## 📊 Key Insights

- Acceleration and gyro values clearly distinguish walking from running
- Tree-based models performed best without risk of overfitting
- Feature scaling was essential for MLP and Logistic Regression
- Achieved **>99% test accuracy** after tuning and data cleaning

---


