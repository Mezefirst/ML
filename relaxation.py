# Relaxation modeling 

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Sample data collection
data = {
    'time': np.arange(0, 100, 1),
    'clamping_force': np.random.normal(100, 5, 100),
    'temperature': np.random.normal(25, 2, 100)
}
df = pd.DataFrame(data)

# Data preprocessing
df['clamping_force'] = df['clamping_force'].apply(lambda x: x if x > 0 else 0)
df['temperature'] = df['temperature'].apply(lambda x: x if x > 0 else 0)

# Feature engineering
df['force_change'] = df['clamping_force'].diff().fillna(0)
df['temp_change'] = df['temperature'].diff().fillna(0)

# Split data into training and testing sets
X = df[['time', 'force_change', 'temp_change']]
y = df['clamping_force']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the ML model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')

# Perform k-fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)
mae_scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=kf)
rmse_scores = cross_val_score(model, X, y, scoring='neg_root_mean_squared_error', cv=kf)
r2_scores = cross_val_score(model, X, y, scoring='r2', cv=kf)

print(f'MAE: {np.mean(-mae_scores)}')
print(f'RMSE: {np.mean(-rmse_scores)}')
print(f'R-squared: {np.mean(r2_scores)}')

# Residual analysis
residuals = y_test - y_pred
plt.scatter(y_pred, residuals)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()
