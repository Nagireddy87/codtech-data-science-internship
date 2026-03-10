# Task 1: Data Pipeline Development


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

print("Starting Data Pipeline...")
df = pd.read_csv("Task1_Data_Pipeline/Telco-Customer-Churn.csv")

print("\nDataset Loaded Successfully!")
print(df.head())

print("\nDataset Shape:", df.shape)

print("\nDataset Info:")
print(df.info())

if "customerID" in df.columns:
    df.drop("customerID", axis=1, inplace=True)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

df.fillna(df.mean(numeric_only=True), inplace=True)

label_encoder = LabelEncoder()

for col in df.select_dtypes(include="object").columns:
    df[col] = label_encoder.fit_transform(df[col])

print("\nCategorical columns encoded successfully!")

X = df.drop("Churn", axis=1)
y = df["Churn"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

print("\nData split completed!")
print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)

print("\nProcessed data saved successfully!")

print("\nData Pipeline Execution Completed!")