import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ==========================================
# 1. GENERATE SYNTHETIC DATASET
# ==========================================
np.random.seed(42)
n_samples = 1000

data = {
    'CustomerID': np.arange(1001, 1001 + n_samples),
    'Age': np.random.randint(18, 70, size=n_samples),
    'Tenure_Months': np.random.randint(1, 60, size=n_samples),
    'MonthlyCharges': np.round(np.random.uniform(20.0, 120.0, size=n_samples), 2),
    'ContractType': np.random.choice(['Month-to-Month', 'One-Year', 'Two-Year'], size=n_samples),
    'TechSupport': np.random.choice(['Yes', 'No'], size=n_samples),
    'Churn': np.random.choice([0, 1], size=n_samples, p=[0.75, 0.25])
}

df = pd.DataFrame(data)

# ==========================================
# 2. DATA PREPROCESSING & ENCODING
# ==========================================
# Drop non-informative identifiers
df_clean = df.drop(columns=['CustomerID'])

# Encode categorical variables
le = LabelEncoder()
df_clean['ContractType'] = le.fit_transform(df_clean['ContractType'])
df_clean['TechSupport'] = le.fit_transform(df_clean['TechSupport'])

# Separate Features (X) and Target (y)
X = df_clean.drop(columns=['Churn'])
y = df_clean['Churn']

# Split into Training (80%) and Testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 3. MODEL TRAINING
# ==========================================
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# ==========================================
# 4. EVALUATION
# ==========================================
y_pred = model.predict(X_test_scaled)

print("--- Model Performance ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# ==========================================
# 5. FEATURE IMPORTANCE VISUALIZATION
# ==========================================
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
plt.figure(figsize=(8, 4))
feature_importances.nlargest(5).plot(kind='barh', color='skyblue')
plt.title('Top Factors Influencing Customer Churn')
plt.xlabel('Relative Importance')
plt.tight_layout()
plt.show()