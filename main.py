# ============================================================
# CREDIT CARD DEFAULT PREDICTION USING ANN
# ============================================================

# ------------------------------------------------------------
# 1. IMPORT LIBRARIES
# ------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping


# ------------------------------------------------------------
# 2. REPRODUCIBILITY
# ------------------------------------------------------------

np.random.seed(42)
tf.random.set_seed(42)


# ------------------------------------------------------------
# 3. LOAD DATASET
# ------------------------------------------------------------

df = pd.read_csv("credit_card_default.csv")


# ------------------------------------------------------------
# 4. BASIC DATA INFORMATION
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("CREDIT CARD DEFAULT PREDICTION - ANN")
print("=" * 65)

print(f"\nDataset Shape : {df.shape}")

print("\nMissing Values Before Preprocessing:")
print(df.isnull().sum().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nTarget Distribution:")
print(df["default_payment_next_month"].value_counts())

print("\nTarget Percentage:")
print(
    (df["default_payment_next_month"]
     .value_counts(normalize=True) * 100)
     .round(2)
)


# ------------------------------------------------------------
# 5. HANDLE MISSING VALUES
# ------------------------------------------------------------

categorical_cols = [
    "sex",
    "education",
    "marriage"
]

for col in categorical_cols:
    df[col] = df[col].fillna("Unknown")


# Numerical missing value
df["age"] = df["age"].fillna(df["age"].median())


# ------------------------------------------------------------
# 6. ENCODE PAYMENT STATUS
# ------------------------------------------------------------

payment_mapping = {
    "Payed duly": 0,
    "Payment delayed 1 month": 1,
    "Payment delayed 2 months": 2,
    "Payment delayed 3 months": 3,
    "Payment delayed 4 months": 4,
    "Payment delayed 5 months": 5,
    "Payment delayed 6 months": 6,
    "Payment delayed 7 months": 7,
    "Payment delayed 8 months": 8,
    "Unknown": -1
}

payment_cols = [
    "payment_status_sep",
    "payment_status_aug",
    "payment_status_jul",
    "payment_status_jun",
    "payment_status_may",
    "payment_status_apr"
]

for col in payment_cols:
    df[col] = df[col].map(payment_mapping)


# ------------------------------------------------------------
# 7. REMOVE UNNECESSARY COLUMN
# ------------------------------------------------------------

if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])


# ------------------------------------------------------------
# 8. ONE-HOT ENCODING
# ------------------------------------------------------------

df = pd.get_dummies(
    df,
    columns=["sex", "education", "marriage"],
    dtype=int
)


# ------------------------------------------------------------
# 9. CHECK PREPROCESSED DATA
# ------------------------------------------------------------

print("\n" + "-" * 65)
print("AFTER PREPROCESSING")
print("-" * 65)

print("Missing Values :", df.isnull().sum().sum())
print("Dataset Shape  :", df.shape)


# ------------------------------------------------------------
# 10. SEPARATE FEATURES AND TARGET
# ------------------------------------------------------------

X = df.drop(
    columns=["default_payment_next_month"]
)

y = df["default_payment_next_month"]


print("\nFeatures (X) :", X.shape)
print("Target (y)  :", y.shape)


# ------------------------------------------------------------
# 11. TRAIN + TEST SPLIT
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ------------------------------------------------------------
# 12. TRAIN + VALIDATION SPLIT
# ------------------------------------------------------------

X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.10,
    random_state=42,
    stratify=y_train
)


# ------------------------------------------------------------
# 13. DATASET DISTRIBUTION
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("DATA SPLIT")
print("=" * 65)

print(f"Training   : {X_train_final.shape}")
print(f"Validation : {X_val.shape}")
print(f"Testing    : {X_test.shape}")

print("\nClass Distribution:")
print(
    f"Training   -> "
    f"0: {(y_train_final == 0).sum()} | "
    f"1: {(y_train_final == 1).sum()}"
)

print(
    f"Validation -> "
    f"0: {(y_val == 0).sum()} | "
    f"1: {(y_val == 1).sum()}"
)

print(
    f"Testing    -> "
    f"0: {(y_test == 0).sum()} | "
    f"1: {(y_test == 1).sum()}"
)


# ------------------------------------------------------------
# 14. FEATURE SCALING
# ------------------------------------------------------------

scaler = StandardScaler()

X_train_final_scaled = scaler.fit_transform(
    X_train_final
)

X_val_scaled = scaler.transform(
    X_val
)

X_test_scaled = scaler.transform(
    X_test
)


print("\nScaling:")
print("StandardScaler applied successfully.")


# ------------------------------------------------------------
# 15. CLASS WEIGHTS
# ------------------------------------------------------------

classes = np.unique(y_train_final)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train_final
)

class_weights = dict(
    zip(classes, weights)
)


print("\nClass Weights:")
for class_value, weight in class_weights.items():
    print(
        f"Class {class_value}: {weight:.4f}"
    )


# ------------------------------------------------------------
# 16. BUILD ANN MODEL
# ------------------------------------------------------------

model = Sequential([
    Input(shape=(X_train_final_scaled.shape[1],)),

    Dense(
        64,
        activation="relu"
    ),

    Dense(
        32,
        activation="relu"
    ),

    Dense(
        16,
        activation="relu"
    ),

    Dense(
        1,
        activation="sigmoid"
    )
])


# ------------------------------------------------------------
# 17. COMPILE MODEL
# ------------------------------------------------------------

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# ------------------------------------------------------------
# 18. MODEL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("ANN MODEL")
print("=" * 65)

model.summary()


# ------------------------------------------------------------
# 19. EARLY STOPPING
# ------------------------------------------------------------

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)


# ------------------------------------------------------------
# 20. TRAIN MODEL
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("MODEL TRAINING")
print("=" * 65)

history = model.fit(
    X_train_final_scaled,
    y_train_final,

    validation_data=(
        X_val_scaled,
        y_val
    ),

    epochs=50,
    batch_size=32,

    class_weight=class_weights,

    callbacks=[early_stopping],

    verbose=1
)


# ------------------------------------------------------------
# 21. TRAINING INFORMATION
# ------------------------------------------------------------

best_epoch = np.argmin(
    history.history["val_loss"]
) + 1

best_val_loss = min(
    history.history["val_loss"]
)

best_val_accuracy = max(
    history.history["val_accuracy"]
)

print("\n" + "-" * 65)
print("TRAINING SUMMARY")
print("-" * 65)

print(
    f"Epochs Completed       : {len(history.history['loss'])}"
)

print(
    f"Best Epoch             : {best_epoch}"
)

print(
    f"Best Validation Loss   : {best_val_loss:.4f}"
)

print(
    f"Best Validation Accuracy: "
    f"{best_val_accuracy:.4f}"
)


# ------------------------------------------------------------
# 22. VALIDATION PREDICTIONS
# ------------------------------------------------------------

y_val_prob = model.predict(
    X_val_scaled,
    verbose=0
).ravel()


# ------------------------------------------------------------
# 23. THRESHOLD ANALYSIS ON VALIDATION SET
# ------------------------------------------------------------

thresholds = np.arange(
    0.10,
    0.91,
    0.05
)

threshold_results = []

for threshold in thresholds:

    y_val_pred = (
        y_val_prob >= threshold
    ).astype(int)

    precision = precision_score(
        y_val,
        y_val_pred,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        y_val_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        y_val_pred,
        zero_division=0
    )

    threshold_results.append(
        (
            threshold,
            precision,
            recall,
            f1
        )
    )


print("\n" + "=" * 65)
print("VALIDATION THRESHOLD ANALYSIS")
print("=" * 65)

print(
    f"{'Threshold':<12}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'F1 Score':<12}"
)

print("-" * 48)

for threshold, precision, recall, f1 in threshold_results:

    print(
        f"{threshold:<12.2f}"
        f"{precision:<12.3f}"
        f"{recall:<12.3f}"
        f"{f1:<12.3f}"
    )


# ------------------------------------------------------------
# 24. FIND BEST THRESHOLD
# ------------------------------------------------------------

best_threshold, best_precision, best_recall, best_f1 = max(
    threshold_results,
    key=lambda x: x[3]
)


print("\n" + "-" * 65)

print(
    f"BEST THRESHOLD = {best_threshold:.2f}"
)

print(
    f"Validation Precision = {best_precision:.3f}"
)

print(
    f"Validation Recall    = {best_recall:.3f}"
)

print(
    f"Validation F1 Score  = {best_f1:.3f}"
)


# ------------------------------------------------------------
# 25. FINAL TEST PREDICTIONS
# ------------------------------------------------------------

y_test_prob = model.predict(
    X_test_scaled,
    verbose=0
).ravel()


# Use threshold selected from VALIDATION set
y_test_pred = (
    y_test_prob >= best_threshold
).astype(int)


# ------------------------------------------------------------
# 26. FINAL TEST METRICS
# ------------------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_test_pred
)

precision = precision_score(
    y_test,
    y_test_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_test_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_test_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_test_prob
)

average_precision = average_precision_score(
    y_test,
    y_test_prob
)


# ------------------------------------------------------------
# 27. CONFUSION MATRIX
# ------------------------------------------------------------

cm = confusion_matrix(
    y_test,
    y_test_pred
)


print("\n" + "=" * 65)
print("FINAL TEST RESULTS")
print("=" * 65)

print(
    f"Decision Threshold   : {best_threshold:.2f}"
)

print(
    f"Accuracy             : {accuracy:.4f}"
)

print(
    f"Precision            : {precision:.4f}"
)

print(
    f"Recall               : {recall:.4f}"
)

print(
    f"F1 Score             : {f1:.4f}"
)

print(
    f"ROC-AUC              : {roc_auc:.4f}"
)

print(
    f"Average Precision    : {average_precision:.4f}"
)


# ------------------------------------------------------------
# 28. CONFUSION MATRIX
# ------------------------------------------------------------

print("\n" + "-" * 65)
print("CONFUSION MATRIX")
print("-" * 65)

print(cm)

print(
    "\nTN =", cm[0, 0],
    "| FP =", cm[0, 1]
)

print(
    "FN =", cm[1, 0],
    "| TP =", cm[1, 1]
)


# ------------------------------------------------------------
# 29. CLASSIFICATION REPORT
# ------------------------------------------------------------

print("\n" + "-" * 65)
print("CLASSIFICATION REPORT")
print("-" * 65)

print(
    classification_report(
        y_test,
        y_test_pred,
        target_names=[
            "Non-Default (0)",
            "Default (1)"
        ],
        zero_division=0
    )
)


# ============================================================
# 30. VISUALIZATIONS
# ============================================================


# ------------------------------------------------------------
# 30.1 TRAINING VS VALIDATION LOSS
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "Training vs Validation Loss"
)

plt.legend()
plt.grid()

plt.savefig("loss_curve.png", bbox_inches="tight")
plt.close()


# ------------------------------------------------------------
# 30.2 TRAINING VS VALIDATION ACCURACY
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title(
    "Training vs Validation Accuracy"
)

plt.legend()
plt.grid()

plt.savefig("accuracy_curve.png", bbox_inches="tight")
plt.close()


# ------------------------------------------------------------
# 30.3 CONFUSION MATRIX HEATMAP
# ------------------------------------------------------------

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[
        "Non-Default",
        "Default"
    ],
    yticklabels=[
        "Non-Default",
        "Default"
    ]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.title(
    "Confusion Matrix"
)

plt.savefig("confusion_matrix.png", bbox_inches="tight")
plt.close()


# ------------------------------------------------------------
# 30.4 ROC CURVE
# ------------------------------------------------------------

fpr, tpr, roc_thresholds = roc_curve(
    y_test,
    y_test_prob
)

plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    label=f"ANN (AUC = {roc_auc:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title(
    "ROC Curve"
)

plt.legend()
plt.grid()

plt.savefig("roc_curve.png", bbox_inches="tight")
plt.close()


# ------------------------------------------------------------
# 30.5 PRECISION-RECALL CURVE
# ------------------------------------------------------------

pr_precision, pr_recall, pr_thresholds = precision_recall_curve(
    y_test,
    y_test_prob
)

plt.figure(figsize=(8, 6))

plt.plot(
    pr_recall,
    pr_precision,
    label=f"ANN (AP = {average_precision:.3f})"
)

plt.xlabel("Recall")
plt.ylabel("Precision")

plt.title(
    "Precision-Recall Curve"
)

plt.legend()
plt.grid()

plt.savefig("precision_recall_curve.png", bbox_inches="tight")
plt.close()


# ------------------------------------------------------------
# 31. FINAL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("PROJECT SUMMARY")
print("=" * 65)

print("Dataset Size          :", len(df))
print("Input Features        :", X.shape[1])
print("Training Samples      :", len(X_train_final))
print("Validation Samples    :", len(X_val))
print("Testing Samples       :", len(X_test))

print("\nANN Architecture:")
print("Input -> 64 -> 32 -> 16 -> 1")

print("\nTraining:")
print("Optimizer             : Adam")
print("Loss Function         : Binary Crossentropy")
print("Activation             : ReLU + Sigmoid")
print("Batch Size            : 32")
print("Early Stopping        : Yes")

print("\nFinal Performance:")
print(f"Threshold             : {best_threshold:.2f}")
print(f"Accuracy              : {accuracy:.4f}")
print(f"Precision             : {precision:.4f}")
print(f"Recall                : {recall:.4f}")
print(f"F1 Score              : {f1:.4f}")
print(f"ROC-AUC               : {roc_auc:.4f}")
print(f"Average Precision     : {average_precision:.4f}")

print("\n" + "=" * 65)
print("END OF EXPERIMENT")
print("=" * 65)