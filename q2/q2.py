import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt


dataset = pd.read_csv("../data/csv_result-dataset_37_diabetes.csv")
dataset = dataset.drop(columns=["id"])
numerical_features = ["preg", "plas", "pres", "skin", "insu", "mass", "pedi", "age"]

# encode the tagets using 0 and 1
label_encoder = LabelEncoder()
dataset["class"] = label_encoder.fit_transform(dataset["class"])

# split the data into train and test sets
X = dataset[numerical_features]
y = dataset["class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# standardize the input features
scaler = StandardScaler()

X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])

# train two classifiers
adaboost_classifier = AdaBoostClassifier(algorithm="SAMME")
adaboost_classifier.fit(X_train, y_train)
y_probs_adaboost = adaboost_classifier.predict_proba(X_test)[:, 1]

lr_classifier = LogisticRegression()
lr_classifier.fit(X_train, y_train)
y_probs_lr = lr_classifier.predict_proba(X_test)[:, 1]

# Create the ROC figure
fpr_adaboost, tpr_adaboost, _ = roc_curve(y_test, y_probs_adaboost)
roc_auc_adaboost = auc(fpr_adaboost, tpr_adaboost)

# Generate ROC curve values for Logistic Regression
fpr_logistic, tpr_logistic, _ = roc_curve(y_test, y_probs_lr)
roc_auc_logistic = auc(fpr_logistic, tpr_logistic)

# Plot the ROC curves
plt.figure(figsize=(10, 6))
plt.plot(fpr_adaboost, tpr_adaboost, label=f'AdaBoost (AUC = {roc_auc_adaboost:.2f})')
plt.plot(fpr_logistic, tpr_logistic, label=f'Logistic Regression (AUC = {roc_auc_logistic:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.savefig('q2-2(roc).png', dpi=300, bbox_inches='tight')
plt.show()

# plot the PR curves

# Generate PR curve values for AdaBoost
precision_adaboost, recall_adaboost, _ = precision_recall_curve(y_test, y_probs_adaboost)
pr_auc_adaboost = average_precision_score(y_test, y_probs_adaboost)

# Generate PR curve values for Logistic Regression
precision_logistic, recall_logistic, _ = precision_recall_curve(y_test, y_probs_lr)
pr_auc_logistic = average_precision_score(y_test, y_probs_lr)

# Plot the PR curves
plt.figure(figsize=(10, 6))
plt.plot(recall_adaboost, precision_adaboost, label=f'AdaBoost (AUC = {pr_auc_adaboost:.2f})')
plt.plot(recall_logistic, precision_logistic, label=f'Logistic Regression (AUC = {pr_auc_logistic:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.savefig('q2-2(pr).png', dpi=300, bbox_inches='tight')
plt.show()


# Generate PR-Gain curve values for Logistic Regression
pi = 0.36


def calculate_auprg(y_true, y_prob, pi):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)

    num_positives = np.sum(y_true == 1)
    num_negatives = np.sum(y_true == 0)

    TP = tpr * num_positives
    TN = (1 - fpr) * num_negatives
    FP = fpr * num_negatives
    FN = num_positives - TP

    raw_precision = np.array([min(1, max(0, x)) for x in FP/TP])
    raw_recall = np.array([min(1, max(0, x)) for x in FN/TP])

    precision_gain = 1 - (pi / (1 - pi)) * raw_precision
    recall_gain = 1 - (pi / (1 - pi)) * raw_recall

    return auc(recall_gain, precision_gain)

print("AUPRG for AdaBoost: ", calculate_auprg(y_test, y_probs_adaboost, 0.36))
print("AUPRG for Logistic Regression: ", calculate_auprg(y_test, y_probs_lr, 0.36))




