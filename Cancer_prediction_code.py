# ============================================================
# BREAST CANCER PREDICTION USING LOGISTIC REGRESSION
# ============================================================


# 1. IMPORT LIBRARIES


import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# 2. LOAD DATASET


df = pd.read_csv("../../data.csv")

print("\n" + "=" * 60)
print(" Breast_Cancer_Dataset")
print("=" * 60)

print("\nFirst 5 Rows:")
print(df.head())

# 3. BASIC EDA


print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)


# 4. MISSING VALUES DATAFRAME


missing_df = pd.DataFrame({
    "Column": df.columns,
    "Missing Values": df.isnull().sum().values
})

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

print(missing_df)


# 5. DIAGNOSIS DISTRIBUTION


diagnosis_df = (
    df["diagnosis"]
    .value_counts()
    .reset_index()
)

diagnosis_df.columns = [
    "Diagnosis",
    "Count"
]

print("\n" + "=" * 60)
print("DIAGNOSIS DISTRIBUTION")
print("=" * 60)

print(diagnosis_df)


# 6. DROP UNNECESSARY COLUMNS




if "id" in df.columns:
    df = df.drop("id", axis=1)



if "Unnamed: 32" in df.columns:
    df = df.drop("Unnamed: 32", axis=1)

print("\n" + "=" * 60)
print("DATASET AFTER DROPPING UNNECESSARY COLUMNS")
print("=" * 60)

print(df.head())

print("\nNew Shape:")
print(df.shape)


# 7. ENCODE TARGET COLUMN


# M = Malignant = 1
# B = Benign = 0

df["diagnosis"] = df["diagnosis"].map({
    "M": 1,
    "B": 0
})

print("\n" + "=" * 60)
print("ENCODED DIAGNOSIS")
print("=" * 60)

print(df["diagnosis"].value_counts())


# 8. SEPARATE FEATURES AND TARGET



X = df.drop("diagnosis", axis=1)


y = df["diagnosis"]


print("\n" + "=" * 60)
print("FEATURES AND TARGET")
print("=" * 60)

print("\nNumber of Features:")
print(X.shape[1])

print("\nFeature Names:")
print(X.columns.tolist())

print("\nTarget:")
print("0 = B - Benign")
print("1 = M - Malignant")


# 9. TRAIN TEST SPLIT


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n" + "=" * 60)
print("TRAIN TEST SPLIT")
print("=" * 60)

print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)


# 10. FEATURE SCALING


scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

print("\n" + "=" * 60)
print("FEATURE SCALING COMPLETED")
print("=" * 60)


# 11. TRAIN LOGISTIC REGRESSION MODEL


model = LogisticRegression(
    max_iter=1000
)

model.fit(
    X_train_scaled,
    y_train
)

print("\n" + "=" * 60)
print("MODEL TRAINING COMPLETED")
print("=" * 60)


# 12. MAKE PREDICTIONS


y_pred = model.predict(
    X_test_scaled
)


# 13. MODEL EVALUATION


accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")


# 14. EVALUATION METRICS DATAFRAME


metrics_df = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ],

    "Score": [
        accuracy,
        precision,
        recall,
        f1
    ]
})

print("\n" + "=" * 60)
print("MODEL METRICS DATAFRAME")
print("=" * 60)

print(metrics_df)


# 15. CONFUSION MATRIX


cm = confusion_matrix(
    y_test,
    y_pred
)

confusion_df = pd.DataFrame(
    cm,

    index=[
        "Actual B - Benign",
        "Actual M - Malignant"
    ],

    columns=[
        "Predicted B - Benign",
        "Predicted M - Malignant"
    ]
)

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(confusion_df)


# 16. CLASSIFICATION REPORT


print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Benign",
            "Malignant"
        ]
    )
)


# 17. FINAL PREDICTION RESULTS DATAFRAME


results_df = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})



results_df["Actual Diagnosis"] = results_df["Actual"].map({
    1: "M - Malignant",
    0: "B - Benign"
})

results_df["Predicted Diagnosis"] = results_df["Predicted"].map({
    1: "M - Malignant",
    0: "B - Benign"
})



results_df["Result"] = results_df.apply(
    lambda row: (
        "Correct"
        if row["Actual"] == row["Predicted"]
        else "Wrong"
    ),
    axis=1
)



results_df = results_df[
    [
        "Actual Diagnosis",
        "Predicted Diagnosis",
        "Result"
    ]
]

print("\n" + "=" * 60)
print("FINAL PREDICTION RESULTS")
print("=" * 60)

print(results_df.head(20))


# 18. SAVE RESULTS TO CSV


results_df.to_csv(
    "cancer_prediction_results.csv",
    index=False
)



print("\nFinal prediction results saved as:")
print("cancer_prediction_results.csv")