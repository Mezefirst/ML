# Exampel of Relaxation preformance of fasteners

# Setup of labiraries 
import pandas as pd
import numpy as np

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

# ML modeling development

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

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

# Model selection 

from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_absolute_error')
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
print("Best parameters:", grid_search.best_params_)

# Ensemble methods 
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import VotingRegressor

model1 = RandomForestRegressor(n_estimators=100, random_state=42)
model2 = GradientBoostingRegressor(n_estimators=100, random_state=42)

ensemble_model = VotingRegressor([('rf', model1), ('gb', model2)])
ensemble_model.fit(X_train, y_train)

# Ensuring model accuracy  
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Perform k-fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)
mae_scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=kf)
rmse_scores = cross_val_score(model, X, y, scoring='neg_root_mean_squared_error', cv=kf)
r2_scores = cross_val_score(model, X, y, scoring='r2', cv=kf)

print(f'MAE: {np.mean(-mae_scores)}')
print(f'RMSE: {np.mean(-rmse_scores)}')
print(f'R-squared: {np.mean(r2_scores)}')

# Regularization 
from sklearn.linear_model import Lasso

lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
y_pred = lasso.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print("MAE with Lasso:", mae)


# Residual analysis
y_pred = model.predict(X_test)
residuals = y_test - y_pred
plt.scatter(y_pred, residuals)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()


# Integraion with standardize protocol
# Function to apply the standardized protocol and ML model
def measure_relaxation_performance(df, model):
    df['predicted_force'] = model.predict(df[['time', 'force_change', 'temp_change']])
    return df

# Apply the protocol and ML model
final_df = measure_relaxation_performance(df, model)
print(final_df.head())

# User-friendly interface
import streamlit as st

# Streamlit app for user-friendly interface
def app():
    st.title("Relaxation Performance Measurement Tool")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        final_df = measure_relaxation_performance(df, model)
        st.write(final_df)
        st.write("Mean Absolute Error:", mae)

if __name__ == "__main__":
    app()
