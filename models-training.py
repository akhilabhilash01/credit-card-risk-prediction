import kagglehub
from kagglehub import KaggleDatasetAdapter

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

import pandas as pd
import joblib
import os
import json

# Load datasets
app_df = pd.read_csv("data/Credit_card.csv")
label_df = pd.read_csv("data/Credit_card_label.csv")

df = app_df.merge(label_df, on="Ind_ID", how="inner")

# Remove missing values
df = df.dropna()

# Encode categorical columns
df = pd.get_dummies(df, drop_first=True)

# Split features and target
X = df.drop("label", axis=1)
y = df["label"]

# Save as json
os.makedirs("model", exist_ok=True)
with open("model/feature_columns.json", "w") as f:
    json.dump(list(X.columns), f)
    
# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

"""1. Logistic Regression"""
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_scaled, y_train)

y_pred_lr = lr_model.predict(X_test_scaled)
y_prob_lr = lr_model.predict_proba(X_test_scaled)[:,1]

# Calculate metrics of Logistic Regression
accuracy_lr = accuracy_score(y_test, y_pred_lr)
auc_lr = roc_auc_score(y_test, y_prob_lr)
precision_lr = precision_score(y_test, y_pred_lr)
recall_lr = recall_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr)
mcc_lr = matthews_corrcoef(y_test, y_pred_lr)

"""2. DECISION TREE CLASSIFIER"""
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

y_pred_dt = dt_model.predict(X_test)
y_prob_dt = dt_model.predict_proba(X_test)[:,1]

# Calculate metrics of Decision Tree Classifier
accuracy_dt = accuracy_score(y_test, y_pred_dt)
auc_dt = roc_auc_score(y_test, y_prob_dt)
precision_dt = precision_score(y_test, y_pred_dt)
recall_dt = recall_score(y_test, y_pred_dt)
f1_dt = f1_score(y_test, y_pred_dt)
mcc_dt = matthews_corrcoef(y_test, y_pred_dt)

"""3. K-Nearest Neighbor Classifier"""
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)

y_pred_knn = knn_model.predict(X_test_scaled)
y_prob_knn = knn_model.predict_proba(X_test_scaled)[:,1]

# Calculate metrics of K-Nearest Neighbor Classifier
accuracy_knn = accuracy_score(y_test, y_pred_knn)
auc_knn = roc_auc_score(y_test, y_prob_knn)
precision_knn = precision_score(y_test, y_pred_knn)
recall_knn = recall_score(y_test, y_pred_knn)
f1_knn = f1_score(y_test, y_pred_knn)
mcc_knn = matthews_corrcoef(y_test, y_pred_knn)


"""4. Gaussian Naive Bayes Classifier"""
gnb_model = GaussianNB()
gnb_model.fit(X_train_scaled, y_train)

y_pred_nb = gnb_model.predict(X_test_scaled)
y_prob_nb = gnb_model.predict_proba(X_test_scaled)[:,1]

# Calculate metrics of Gaussian Naive Bayes Classifier
accuracy_gnb = accuracy_score(y_test, y_pred_nb)
auc_gnb = roc_auc_score(y_test, y_prob_nb)
precision_gnb = precision_score(y_test, y_pred_nb)
recall_gnb = recall_score(y_test, y_pred_nb)
f1_gnb = f1_score(y_test, y_pred_nb)
mcc_gnb = matthews_corrcoef(y_test, y_pred_nb)


"""5. Random Forest"""
rf_model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:,1]

# Calculate metrics of Random Forest
accuracy_rf = accuracy_score(y_test, y_pred_rf)
auc_rf = roc_auc_score(y_test, y_prob_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)
mcc_rf = matthews_corrcoef(y_test, y_pred_rf)

"""6. XGBoost"""
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

xgb_model = XGBClassifier(
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    random_state=42
)

xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)
y_prob_xgb = xgb_model.predict_proba(X_test)[:,1]

# Calculate metrics of XGBoost
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
auc_xgb = roc_auc_score(y_test, y_prob_xgb)
precision_xgb = precision_score(y_test, y_pred_xgb)
recall_xgb = recall_score(y_test, y_pred_xgb)
f1_xgb = f1_score(y_test, y_pred_xgb)
mcc_xgb = matthews_corrcoef(y_test, y_pred_xgb)

"""COMPARISON TABLE"""
comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "kNN",
        "Gaussian Naive Bayes",
        "Random Forest (Ensemble)",
        "XGBoost (Ensemble)"
    ],
    "Accuracy": [
        accuracy_lr, accuracy_dt, accuracy_knn,
        accuracy_gnb, accuracy_rf, accuracy_xgb
    ],
    "AUC": [
        auc_lr, auc_dt, auc_knn,
        auc_gnb, auc_rf, auc_xgb
    ],
    "Precision": [
        precision_lr, precision_dt, precision_knn,
        precision_gnb, precision_rf, precision_xgb
    ],
    "Recall": [
        recall_lr, recall_dt, recall_knn,
        recall_gnb, recall_rf, recall_xgb
    ],
    "F1": [
        f1_lr, f1_dt, f1_knn,
        f1_gnb, f1_rf, f1_xgb
    ],
    "MCC": [
        mcc_lr, mcc_dt, mcc_knn,
        mcc_gnb, mcc_rf, mcc_xgb
    ]
})

comparison.set_index("Model", inplace=True)
# Save to CSV
comparison.round(4).to_csv(f"results/comparison_table.csv")


"""Confusion Matix"""
# Store predictions
predictions = {
    "Logistic Regression": y_pred_lr,
    "Decision Tree": y_pred_dt,
    "kNN": y_pred_knn,
    "Gaussian Naive Bayes": y_pred_nb,
    "Random Forest": y_pred_rf,
    "XGBoost": y_pred_xgb
}


# Save confusion matrices
os.makedirs("results/confusion_matrices", exist_ok=True)
for model_name, y_pred_model in predictions.items():
    cm = confusion_matrix(y_test, y_pred_model)
    cm_df = pd.DataFrame(
        cm,
        index=["Actual Normal", "Actual Fraud"],
        columns=["Predicted Normal", "Predicted Fraud"]
    )
    filename = model_name.replace(" ", "_") + "_confusion_matrix.csv"
    cm_df.to_csv(f"results/confusion_matrices/{filename}")


# Save models and scaler
os.makedirs("model", exist_ok=True)
joblib.dump(lr_model, "model/logistic_regression.pkl")
joblib.dump(dt_model, "model/decision_tree.pkl")
joblib.dump(knn_model, "model/knn.pkl")
joblib.dump(gnb_model, "model/naive_bayes.pkl")
joblib.dump(rf_model, "model/random_forest.pkl")
joblib.dump(xgb_model, "model/xgboost.pkl")
joblib.dump(scaler, "model/scaler.pkl")
