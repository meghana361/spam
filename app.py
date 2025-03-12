import streamlit as st
import pandas as pd
import pickle

# Load the model
@st.cache_resource
def load_model():
    with open('spam_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# Load the dataset (for feature names)
@st.cache_data
def load_data():
    df = pd.read_csv("your_dataset.csv")  # Replace with your dataset path
    return df

df = load_data()

# Separate features (for feature names)
X = df.drop('class', axis=1)

# Streamlit App
st.title("Spam Email Classification")

st.write("Enter email features to predict if it's spam.")

# Create input widgets for each feature
input_features = {}
for col in X.columns:
    input_features[col] = st.number_input(f"Enter {col}", value=0.0)

# Create a DataFrame from user inputs
input_data = pd.DataFrame([input_features])

if st.button("Predict"):
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.write("This email is predicted to be SPAM.")
    else:
        st.write("This email is predicted to be NOT SPAM.")

# Optional: Add Model Evaluation if you want to.
# Note: For this, you will need the test dataset, and the test labels.
# Then you can calculate and display metrics.
# Example if you have X_test and y_test available:
# from sklearn.metrics import accuracy_score, classification_report
# if st.checkbox("Show Model Evaluation"):
#     y_pred = model.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     report = classification_report(y_test, y_pred)
#
#     st.write(f"Accuracy: {accuracy:.4f}")
#     st.text("Classification Report:\n" + report)
