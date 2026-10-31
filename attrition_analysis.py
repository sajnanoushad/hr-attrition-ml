import pandas as pd
import numpy as np

# Set a random seed so our results are reproducible
np.random.seed(42)

# Create a sample dataset of 100 employees
n_employees = 100

data = {
    'employee_id': [f"EMP_{i+1:03d}" for i in range(n_employees)],
    'department': np.random.choice(['IT', 'Finance', 'HR', 'Sales', 'Marketing'], size=n_employees),
    'monthly_salary': np.random.randint(3500, 9500, size=n_employees),
    'years_at_company': np.random.randint(1, 11, size=n_employees),
    'satisfaction_score': np.random.randint(1, 6, size=n_employees),  # Rating from 1 to 5
    'attrition': np.random.choice(['Yes', 'No'], size=n_employees, p=[0.25, 0.75])  # 25% churn rate
}

# Convert dictionary to DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('hr_data.csv', index=False)
print("✅ Dataset successfully created and saved as 'hr_data.csv'!")

# Display the first 5 rows
print("\n--- First 5 Employee Records ---")
print(df.head())
# --- EXPLORATORY DATA ANALYSIS (EDA) ---
print("\n--- Dataset Summary Info ---")
print(df.info())

print("\n--- Average Salary by Department ---")
dept_salary = df.groupby('department')['monthly_salary'].mean()
print(dept_salary)

print("\n--- Attrition Count by Department ---")
attrition_count = pd.crosstab(df['department'], df['attrition'])
print(attrition_count)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# --- MACHINE LEARNING MODELING ---
print("\n--- Training Attrition Prediction Model ---")

# 1. Prepare Features (X) and Target (y)
# We convert the categorical 'department' into numeric columns using get_dummies
df_model = pd.get_dummies(df, columns=['department'], drop_first=True)

X = df_model.drop(columns=['employee_id', 'attrition'])
y = df_model['attrition'] # 'Yes' or 'No'

# 2. Split data into Training and Testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train the Logistic Regression Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 4. Make Predictions and Evaluate
y_pred = model.predict(X_test)

print(f"Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))