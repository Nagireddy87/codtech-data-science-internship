# Task 1: Data Pipeline Development
# CodTech Data Science Internship

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

print("Starting Data Pipeline...")

# 1️⃣ Load Dataset
df = pd.read_csv("Telco-Customer-Churn.csv")

print("\nDataset Loaded Successfully!")
print(df.head())

# 2️⃣ Dataset Information
print("\nDataset Shape:", df.shape)

print("\nDataset Info:")
print(df.info())

# 3️⃣ Remove unnecessary column
if "customerID" in df.columns:
    df.drop("customerID", axis=1, inplace=True)

# 4️⃣ Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# 5️⃣ Handle missing values
df.fillna(df.mean(numeric_only=True), inplace=True)

# 6️⃣ Encode categorical variables
label_encoder = LabelEncoder()

for col in df.select_dtypes(include="object").columns:
    df[col] = label_encoder.fit_transform(df[col])

print("\nCategorical columns encoded successfully!")

# 7️⃣ Split features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# 8️⃣ Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 9️⃣ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

print("\nData split completed!")
print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# 🔟 Save processed data
np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)

print("\nProcessed data saved successfully!")

print("\nData Pipeline Execution Completed!")