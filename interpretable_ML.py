import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv(r"C:\Users\rams6\heart_cleveland_upload.csv")

df = load_data()
st.title("Heart Disease Data Viewer")
st.write(df.head())

# Sidebar
st.sidebar.title("Diabetes Risk Prediction")
model_choice = st.sidebar.selectbox("Select model", ["Logistic Regression", "Decision Tree", "Random Forest"])

st.title("Interpretable Healthcare ML: Diabetes Risk Predictor")
st.markdown("This tool uses machine learning models to predict diabetes risk and explain the results using SHAP.")

# Data Preprocessing
# Display actual column names
st.write("Loaded columns:", df.columns.tolist())

# Use the correct column name for target
TARGET_COL = "condition"  # change this if needed
X = df.drop(TARGET_COL, axis=1)
y = df[TARGET_COL]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Models
log_reg = LogisticRegression(max_iter=1000).fit(X_train_scaled, y_train)
tree = DecisionTreeClassifier(max_depth=4, random_state=42).fit(X_train, y_train)
rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)

# Model selection
if model_choice == "Logistic Regression":
    model = log_reg
    input_data = scaler.transform(X_test)
    test_data = scaler.transform(X)
elif model_choice == "Decision Tree":
    model = tree
    input_data = X_test
    test_data = X
else:
    model = rf
    input_data = X_test
    test_data = X

# Evaluation
y_pred = model.predict(input_data)
st.subheader("Model Performance")
st.write("**Accuracy:**", accuracy_score(y_test, y_pred))
st.write("**Confusion Matrix:**")
st.write(confusion_matrix(y_test, y_pred))
st.write("**Classification Report:**")
st.text(classification_report(y_test, y_pred))

# SHAP Explanation
st.subheader("SHAP Explanations (Global)")

explainer = shap.LinearExplainer(model, X_train, feature_perturbation="interventional")
shap_values = explainer.shap_values(test_data)

fig, ax = plt.subplots(figsize=(10, 5))
shap_values = explainer.shap_values(X_test)

# If shap_values is a list (multi-class or binary classifier), pick one class
if isinstance(shap_values, list):
    shap_values_to_plot = shap_values[1]  # or shap_values[0]
else:
    shap_values_to_plot = shap_values  # already a 2D matrix

shap.summary_plot(shap_values_to_plot, X_test, feature_names=X.columns, plot_type="bar", show=False)
st.pyplot(fig)

# Custom Prediction
st.subheader("Predict on Custom Input")

def user_input_features():
    input_vals = {}
    for col in X.columns:
        val = st.number_input(f"{col}", min_value=0.0, step=1.0)
        input_vals[col] = val
    return pd.DataFrame([input_vals])

user_input = user_input_features()

if st.button("Predict"):
    if model_choice == "Logistic Regression":
        scaled_input = scaler.transform(user_input)
        prediction = model.predict(scaled_input)
        shap_input = scaled_input
    else:
        prediction = model.predict(user_input)
        shap_input = user_input

    st.write(f"**Prediction:** {'Diabetic' if prediction[0] == 1 else 'Non-Diabetic'}")

    st.subheader("SHAP Explanation (Local)")
    shap.initjs()
    shap_vals = explainer.shap_values(shap_input)
    expected_val = explainer.expected_value

    # If shap_vals is a list (multi-class), pick class 1
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
        expected_val = expected_val[1]

    # Now plot
    shap_html = shap.force_plot(expected_val, shap_vals, shap_input, feature_names=X.columns, matplotlib=False)
    st.components.v1.html(shap_html.html(), height=300)

