print("24BAD409-Shalini A")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv(r"C:\Users\SHALINI A\Downloads\LICI - 10 minute data (2).csv")
print("Columns in dataset:", data.columns)
data['Price_Movement'] = np.where(data['close'] > data['open'], 1, 0)
data = data.dropna()
X = data[['open', 'high', 'low', 'volume']]
y = data['Price_Movement']
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
logreg = LogisticRegression(max_iter=5000)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
y_prob = logreg.predict_proba(X_test)[:,1]
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-Score:", f1_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
feature_importance = pd.DataFrame({
    'Feature': ['open', 'high', 'low', 'volume'],
    'Coefficient': logreg.coef_[0]
})
print("Feature Importance:\n", feature_importance.sort_values(by='Coefficient', ascending=False))
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],       
    'solver': ['saga'],            
    'max_iter': [5000]
}
grid = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='f1', n_jobs=-1)
grid.fit(X_train, y_train)
print("Best Hyperparameters:", grid.best_params_)
best_logreg = grid.best_estimator_
y_pred_best = best_logreg.predict(X_test)
print("Final Model Accuracy:", accuracy_score(y_test, y_pred_best))
print("Final Model Precision:", precision_score(y_test, y_pred_best))
print("Final Model Recall:", recall_score(y_test, y_pred_best))
print("Final Model F1-Score:", f1_score(y_test, y_pred_best))
print("Final Confusion Matrix:\n", confusion_matrix(y_test, y_pred_best))
