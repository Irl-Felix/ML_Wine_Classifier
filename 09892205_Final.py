from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# Load the dataset
file_path = 'winequality-white.csv'
wine_data = pd.read_csv(file_path)

# Separating the features and the target variable
X = wine_data.drop('quality', axis=1)
y = wine_data['quality']

# Scaling the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# Initializing 
rf = RandomForestClassifier(random_state=42, class_weight='balanced')
svm = SVC(class_weight='balanced')
log_reg = LogisticRegression(class_weight='balanced')

# Training the models
rf.fit(X_train, y_train)
svm.fit(X_train, y_train)
log_reg.fit(X_train, y_train)

# Making predictions and evaluating the models
rf_pred = rf.predict(X_test)
svm_pred = svm.predict(X_test)
log_reg_pred = log_reg.predict(X_test)

# Classification reports
rf_report = classification_report(y_test, rf_pred)
svm_report = classification_report(y_test, svm_pred)
log_reg_report = classification_report(y_test, log_reg_pred)


#################################
#################################

# Saving
joblib.dump(rf, 'random_forest_model.pkl')
joblib.dump(svm, 'svm_model.pkl')
joblib.dump(log_reg, 'log_reg_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

#Loading the models 

rf_model = joblib.load('random_forest_model.pkl')
svm_model = joblib.load('svm_model.pkl')
log_reg_model = joblib.load('log_reg_model.pkl')
scaler = joblib.load('scaler.pkl')


# Function to make predictions
def make_predictions(input_data):
    scaled_data = scaler.transform(np.array(input_data).reshape(1, -1))
    return {
        "Random Forest": rf_model.predict(scaled_data)[0],
        "SVM": svm_model.predict(scaled_data)[0],
        "Logistic Regression": log_reg_model.predict(scaled_data)[0]
    }

######  The interface of the app
st.title("White Wine Quality Prediction")

# User inputs
st.sidebar.header("Input Wine Features")
fixed_acidity = st.sidebar.slider("Fixed Acidity", min_value=3.0, max_value=15.0, value=7.0)
volatile_acidity = st.sidebar.slider("Volatile Acidity", min_value=0.0, max_value=1.5, value=0.3)
citric_acid = st.sidebar.slider("Citric Acid", min_value=0.0, max_value=1.0, value=0.3)
residual_sugar = st.sidebar.slider("Residual Sugar", min_value=0.0, max_value=70.0, value=6.0)
chlorides = st.sidebar.slider("Chlorides", min_value=0.0, max_value=0.5, value=0.05)
free_sulfur_dioxide = st.sidebar.slider("Free Sulfur Dioxide", min_value=0, max_value=300, value=30)
total_sulfur_dioxide = st.sidebar.slider("Total Sulfur Dioxide", min_value=0, max_value=500, value=100)
density = st.sidebar.slider("Density", min_value=0.98, max_value=1.04, value=0.99)
pH = st.sidebar.slider("pH", min_value=2.5, max_value=4.0, value=3.2)
sulphates = st.sidebar.slider("Sulphates", min_value=0.0, max_value=2.0, value=0.5)
alcohol = st.sidebar.slider("Alcohol", min_value=8.0, max_value=15.0, value=10.0)


if st.sidebar.button("Predict Quality"):
    input_data = [fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
                  free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]
    
    predictions = make_predictions(input_data)

    st.subheader("Predicted Quality")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Random Forest**")
        st.success(f"Quality: {predictions['Random Forest']}")

    with col2:
        st.markdown("**SVM**")
        st.success(f"Quality: {predictions['SVM']}")

    with col3:
        st.markdown("**Logistic Regression**")
        st.success(f"Quality: {predictions['Logistic Regression']}")


st.subheader("Data Stats")
st.write(wine_data.describe())

#  Matrix
st.subheader("Correlation Matrix")
plt.figure(figsize=(12, 8))
sns.heatmap(wine_data.corr(), annot=True, fmt=".2f", cmap="coolwarm")
st.pyplot(plt)


selected_features = ['density', 'pH', 'alcohol']

#  Distributions
st.subheader("Feature Distributions")
fig, axes = plt.subplots(1, len(selected_features), figsize=(15, 4))  
for i, feature in enumerate(selected_features):
    sns.histplot(wine_data[feature], kde=True, bins=30, ax=axes[i])
    axes[i].set_title(feature.capitalize())
st.pyplot(fig)

# Box Plots 
st.subheader("Box Plots for Selected Features")
fig, axes = plt.subplots(1, len(selected_features), figsize=(15, 4))  
for i, feature in enumerate(selected_features):
    sns.boxplot(y=wine_data[feature], ax=axes[i])
    axes[i].set_title(feature.capitalize())
st.pyplot(fig)

# Display Classification Reports
st.subheader("Classification Reports")

st.markdown("**Random Forest Classifier Report:**")
st.text(rf_report)

st.markdown("**Support Vector Machine Classifier Report:**")
st.text(svm_report)

st.markdown("**Logistic Regression Classifier Report:**")
st.text(log_reg_report)