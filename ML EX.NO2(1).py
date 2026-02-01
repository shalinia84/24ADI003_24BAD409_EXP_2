print("24BAD409-Shalini A")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
df = pd.read_csv(
    r"C:\Users\SHALINI A\Downloads\bottle.csv (1).zip",
    low_memory=False
)
df.columns = df.columns.str.strip().str.lower()
column_map = {
    "lat_dec": "latitude",
    "latitude": "latitude",
    "lon_dec": "longitude",
    "longitude": "longitude"
}
df.rename(columns=column_map, inplace=True)
base_features = ['depthm', 'salnty', 'o2ml_l']
optional_features = []
if 'latitude' in df.columns:
    optional_features.append('latitude')
if 'longitude' in df.columns:
    optional_features.append('longitude')
features = base_features + optional_features
target = 't_degc'
data = df[features + [target]]
data = data.fillna(data.mean())
X = data[features]
y = data[target]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
mse = mean_squared_error(y_test, y_pred_lr)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_lr)
print("\nLinear Regression Results")
print("-------------------------")
print("MSE :", mse)
print("RMSE:", rmse)
print("R²  :", r2)
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred_lr, alpha=0.5)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Temperature (°C)")
plt.ylabel("Predicted Temperature (°C)")
plt.title("Actual vs Predicted Temperature")
plt.show()
residuals = y_test - y_pred_lr
plt.figure(figsize=(6,4))
sns.histplot(residuals, kde=True)
plt.title("Residual Error Distribution")
plt.xlabel("Residual Error")
plt.show()
plt.figure(figsize=(8,6))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)
print("\nRidge Regression")
print("RMSE:", np.sqrt(mean_squared_error(y_test, ridge_pred)))
print("R²  :", r2_score(y_test, ridge_pred))
lasso = Lasso(alpha=0.01)
lasso.fit(X_train, y_train)
lasso_pred = lasso.predict(X_test)
print("\nLasso Regression")
print("RMSE:", np.sqrt(mean_squared_error(y_test, lasso_pred)))
print("R²  :", r2_score(y_test, lasso_pred))
