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
