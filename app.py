import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Increase Pandas Styler render limit
pd.set_option("styler.render.max_elements", 500000)  # Adjust based on your dataset size

# Load the model
with open('./Regression_models.pkl', 'rb') as f:
    models = pickle.load(f)

#streamlit title and description
st.title("Human Activity Recognition")
st.write("This app predicts HUMAN activities based on sensor data.")

# Input widgets
st.sidebar.header("Upload & Configuration")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

model_options = ["Logistic Regression", "LSTM", "CNN", "All"]
selected_model = st.sidebar.selectbox("Choose Model(s) to Run", model_options)

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        # Activity label mapping
        activity_map = {
            0: 'STANDING',
            1: 'SITTING',
            2: 'LAYING',
            3: 'WALKING',
            4: 'WALKING_DOWNSTAIRS',
            5: 'WALKING_UPSTAIRS'
        }

        st.dataframe(data)
        actual_activity_col = "Activity"  
        features = data.drop(columns=[actual_activity_col])  # Drop actual activity for feature extraction
        scale = StandardScaler()
        scaled_data = scale.fit_transform(features)

        # Predictions
        if selected_model in ["Logistic Regression", "All"]:
            lr_predictions = models['LogisticRegression'].predict(scaled_data)
            data['LogisticRegression_Prediction'] = lr_predictions

        if selected_model in ["LSTM", "All"]:
            ls_pred_probs = models['LSTM'].predict(scaled_data)
            lstm_predictions = np.array([np.argmax(probs) for probs in ls_pred_probs])
            # Resolve activity labels
            lstm_activity_predictions = [activity_map[pred] for pred in lstm_predictions]
            data['LSTM_Prediction'] = lstm_activity_predictions

        if selected_model in ["CNN", "All"]:
            cnn_pred_probs = models['CNN'].predict(scaled_data)
            cnn_predictions = np.array([np.argmax(probs) for probs in cnn_pred_probs])
            # Resolve activity labels
            cnn_activity_predictions = [activity_map[pred] for pred in cnn_predictions]
            data['CNN_Prediction'] = cnn_activity_predictions

        # Display predictions
        st.write("🔍 **Predictions:**")
        st.dataframe(data)

    except Exception as e:
        st.error(f"Error processing data: {e}")
