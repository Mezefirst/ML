#Sample Process for Using ML to Predict and Optimize Cold Working Degree
"""
Here's a step-by-step guide to collecting data, training ML models, and using the predictions to optimize the cold working process:

1. Collect Data from Incremental Cold Working Process
Data Collection Steps:

Prepare Samples: Collect samples of the material and apply incremental degrees of cold working (e.g., 10%, 20%, 30%, etc.).

Measure Properties: For each increment, measure relevant properties such as:

Degree of cold working (percentage deformation)

Tensile strength

Hardness

Residual stress

Relaxation performance (e.g., clamping force over time)"""

# Sample of cold working degree for controlling relaxation performance 
# Data collection 

import pandas as pd

data = {
    'cold_work_degree': [10, 20, 30, 40, 50],
    'tensile_strength': [400, 450, 500, 520, 530],
    'hardness': [200, 220, 250, 260, 265],
    'residual_stress': [50, 60, 70, 80, 85],
    'relaxation_performance': [0.9, 0.85, 0.8, 0.78, 0.75]  # Higher values indicate better performance
}
df = pd.DataFrame(data)
print(df)

# Train ML medels on collected data
# Data preprocessing 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Split the data into features (X) and target (y)
X = df[['cold_work_degree', 'tensile_strength', 'hardness', 'residual_stress']]
y = df['relaxation_performance']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the ML model
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Initialize and train the Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Predict and evaluate the model
y_pred = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')

# Use predictions to fine-tune the cold working process
# Predict Optimal cold working Degree
# Create a function to predict relaxation performance for a given degree of cold working
def predict_relaxation_performance(cold_work_degree, tensile_strength, hardness, residual_stress):
    features = scaler.transform([[cold_work_degree, tensile_strength, hardness, residual_stress]])
    return model.predict(features)[0]

# Example prediction
optimal_cold_work_degree = 35  # Hypothetical optimal degree
predicted_performance = predict_relaxation_performance(optimal_cold_work_degree, 480, 240, 75)
print(f'Predicted Relaxation Performance for {optimal_cold_work_degree}% Cold Work: {predicted_performance}')

#Fine-Tune Process:

"""Based on the predictions, adjust the cold working process parameters to achieve the desired balance between strength and relaxation performance.

Continuously collect data and retrain the ML model to improve its accuracy and reliability over time.

By following these steps, you can effectively use ML to predict the optimal degree of deformation and fine-tune the cold working process to achieve a balance between strength and relaxation performance."""