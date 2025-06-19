import streamlit as st
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

st.title("🌸 Iris Flower Prediction App")

st.write("""
Enter flower features below to predict the species.
""")

# Sidebar input
st.sidebar.header("Input Features")

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length (cm)', 4.3, 7.9, 5.1)
    sepal_width = st.sidebar.slider('Sepal width (cm)', 2.0, 4.4, 3.5)
    petal_length = st.sidebar.slider('Petal length (cm)', 1.0, 6.9, 1.4)
    petal_width = st.sidebar.slider('Petal width (cm)', 0.1, 2.5, 0.2)
    data = {
        'sepal length (cm)': sepal_length,
        'sepal width (cm)': sepal_width,
        'petal length (cm)': petal_length,
        'petal width (cm)': petal_width
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Load model and dataset
iris = load_iris()
X = iris.data
y = iris.target

model = RandomForestClassifier()
model.fit(X, y)

# Make prediction
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

# Show results
st.subheader("Prediction")
st.write(f"Predicted Species: **{iris.target_names[prediction[0]]}**")

st.subheader("Prediction Probability")
st.write(pd.DataFrame(prediction_proba, columns=iris.target_names))
