import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the model
#model = pickle.load(open("Logistic_Regression.pkl", 'rb'))
with open('Regression_models.pkl', 'rb') as f:
    models = pickle.load(f)

#streamlit title and description
st.title("Human Activity Recognition")
st.write("This app predicts human activities based on sensor data.")

# Input widgets
st.sidebar.header("Input Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

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

        # Predictions
        lr_predictions = models['LogisticRegression'].predict(data)

       # ff_pred_probs = models['FeedForward'].predict(data)
       # feedforward_predictions = np.array([np.argmax(probs) for probs in ff_pred_probs])
        # Resolve activity labels
       # ff_activity_predictions = [activity_map[pred] for pred in feedforward_predictions]

        ls_pred_probs = models['LSTM'].predict(data)
        lstm_predictions = np.array([np.argmax(probs) for probs in ls_pred_probs])
        # Resolve activity labels
        lstm_activity_predictions = [activity_map[pred] for pred in lstm_predictions]

        cnn_pred_probs = models['CNN'].predict(data)
        cnn_predictions = np.array([np.argmax(probs) for probs in cnn_pred_probs])
        # Resolve activity labels
        cnn_activity_predictions = [activity_map[pred] for pred in cnn_predictions]
        
        # Add predictions to the DataFrame
        data['LogisticRegression_Prediction'] = lr_predictions
       # data['FeedForward'] = feedforward_predictions
        data['LSTM_Prediction'] = lstm_activity_predictions
        data['CNN_Prediction'] = cnn_activity_predictions

        # Display predictions
        st.write("🔍 **Predictions:**")
        st.dataframe(data)

    except Exception as e:
        st.error(f"Error processing data: {e}")