# 📊 HR Analytics & Employee Attrition Predictor

An end-to-end Data Science and Machine Learning project that analyzes synthetic HR employee records to identify key drivers of employee turnover and predicts attrition using Logistic Regression.

---

## 📌 Project Overview
Employee turnover can be costly for organizations. This project demonstrates a complete data workflow—from synthetic data generation and Exploratory Data Analysis (EDA) to building a Machine Learning classification model to predict whether an employee is likely to leave (`Yes` / `No`).

## 🛠️ Tech Stack & Libraries
* **Language:** Python 3.x
* **Data Manipulation:** `pandas`, `numpy`
* **Machine Learning:** `scikit-learn` (Logistic Regression, Train-Test Split, Metrics)
* **IDE / Editor:** Cursor AI / VS Code

---

## 💡 Key Features & Workflow
1. **Data Generation:** Created a reproducible dataset of 100 employee records with attributes like `monthly_salary`, `years_at_company`, `satisfaction_score`, and `department`.
2. **Exploratory Data Analysis (EDA):**
   * Aggregated average monthly salary across departments.
   * Calculated attrition cross-tabulations to inspect high-churn departments.
3. **Predictive Modeling:**
   * One-hot encoded categorical variables (`department`).
   * Split data into 80% training and 20% testing sets.
   * Trained a Logistic Regression model achieving **75% accuracy** on unseen test data.

---

## 🚀 How to Run the Project

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR-USERNAME/hr-attrition-ml.git](https://github.com/YOUR-USERNAME/hr-attrition-ml.git)
   cd hr-attrition-ml