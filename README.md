# Credit Card Default Prediction using ANN 💳🤖

An end-to-end Deep Learning (Artificial Neural Network) project to predict credit card client defaults for the upcoming month using customer demographic, financial, and payment history data.

---

## 📌 Project Overview

Defaulting on credit card payments poses a major financial risk to credit institutions and banks. Predicting potential defaults allows financial providers to proactively manage credit limits, manage risk exposure, and optimize collections.

This repository implements a complete Machine Learning / Deep Learning pipeline using **TensorFlow / Keras** and **scikit-learn** to analyze, preprocess, train, optimize, and evaluate a multi-layer Neural Network on credit card client data.

---

## 📊 Dataset Overview

- **Source File**: `credit_card_default.csv`
- **Total Records**: 30,000 clients
- **Target Variable**: `default_payment_next_month`
  - `0`: Non-Default (77.88%)
  - `1`: Default (22.12%)

### Key Features:
- **Demographics**: `LIMIT_BAL`, `sex`, `education`, `marriage`, `age`
- **Repayment Status (Apr - Sep)**: `payment_status_sep`, `payment_status_aug`, `payment_status_jul`, `payment_status_jun`, `payment_status_may`, `payment_status_apr`
- **Bill Amounts (Apr - Sep)**: `bill_amt_sep`, `bill_amt_aug`, `bill_amt_jul`, `bill_amt_jun`, `bill_amt_may`, `bill_amt_apr`
- **Previous Payment Amounts (Apr - Sep)**: `pay_amt_sep`, `pay_amt_aug`, `pay_amt_jul`, `pay_amt_jun`, `pay_amt_may`, `pay_amt_apr`

---

## 🛠️ Pipeline & Architecture

### 1. Preprocessing & Feature Engineering
- **Missing Value Handling**: Median imputation for numerical fields (`age`), custom categorical category (`"Unknown"`).
- **Payment Status Mapping**: Ordinal mapping of repayment status delay values.
- **Categorical One-Hot Encoding**: Converted `sex`, `education`, and `marriage` into binary dummy variables.
- **Data Splitting**:
  - Training Set: 21,600 samples (72%)
  - Validation Set: 2,400 samples (8%)
  - Testing Set: 6,000 samples (20% Stratified)
- **Feature Scaling**: Applied `StandardScaler` fitted on training data.
- **Class Imbalance Management**: Applied balanced `class_weights` during model training (`Class 0: 0.6420`, `Class 1: 2.2604`).

### 2. ANN Model Architecture
- **Input Layer**: 32 Features
- **Hidden Layer 1**: 64 Units (ReLU activation)
- **Hidden Layer 2**: 32 Units (ReLU activation)
- **Hidden Layer 3**: 16 Units (ReLU activation)
- **Output Layer**: 1 Unit (Sigmoid activation)

```
Input (32) ──> Dense(64, ReLU) ──> Dense(32, ReLU) ──> Dense(16, ReLU) ──> Dense(1, Sigmoid)
```

### 3. Optimization & Training
- **Optimizer**: Adam
- **Loss Function**: Binary Crossentropy
- **Callbacks**: Early Stopping (`monitor='val_loss'`, `patience=5`)
- **Batch Size**: 32
- **Epochs**: Up to 50

---

## 🏆 Model Performance & Results

### Final Test Set Metrics (at Decision Threshold = 0.60):

| Metric | Score |
| :--- | :--- |
| **Accuracy** | `78.50%` |
| **Precision** | `51.31%` |
| **Recall** | `54.41%` |
| **F1 Score** | `0.5282` |
| **ROC-AUC** | `0.7576` |
| **PR-AUC (Average Precision)** | `0.5263` |

### Confusion Matrix:

```
                  Predicted Non-Default (0)    Predicted Default (1)
Actual Non-Default (0)       3988 (TN)                    685 (FP)
Actual Default (1)            605 (FN)                    722 (TP)
```

---

## 📈 Visualizations

The pipeline generates training and evaluation plots saved as PNG images:

1. **Training vs. Validation Loss (`loss_curve.png`)**
2. **Training vs. Validation Accuracy (`accuracy_curve.png`)**
3. **Confusion Matrix Heatmap (`confusion_matrix.png`)**
4. **ROC Curve (`roc_curve.png`)**
5. **Precision-Recall Curve (`precision_recall_curve.png`)**

---

## 📁 Repository Structure

```
credit-card-default-prediction/
├── credit_card_default.csv     # Dataset file
├── dataset.py                  # Script to download dataset via kagglehub
├── main.py                     # Main execution pipeline
├── README.md                   # Project documentation
├── .gitignore                  # Git ignore rules
├── loss_curve.png              # Loss curve plot
├── accuracy_curve.png          # Accuracy curve plot
├── confusion_matrix.png        # Confusion matrix plot
├── roc_curve.png               # ROC curve plot
└── precision_recall_curve.png  # Precision-Recall curve plot
```

---

## 🚀 How to Run

### 1. Prerequisites
Ensure Python 3.10+ is installed along with dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow kagglehub
```

### 2. Run the Pipeline
Execute `main.py`:

```bash
python main.py
```

---

## 📝 License

This project is open-source and available under the MIT License.
