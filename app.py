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
        actual_activity_col = "Activity"  
        features = data.drop(columns=[actual_activity_col])  # Drop actual activity for feature extraction
        scale = StandardScaler()
        scaled_data = scale.fit_transform(features)

        # Predictions
        lr_predictions = models['LogisticRegression'].predict(scaled_data)

        ls_pred_probs = models['LSTM'].predict(scaled_data)
        lstm_predictions = np.array([np.argmax(probs) for probs in ls_pred_probs])
        # Resolve activity labels
        lstm_activity_predictions = [activity_map[pred] for pred in lstm_predictions]

        cnn_pred_probs = models['CNN'].predict(scaled_data)
        cnn_predictions = np.array([np.argmax(probs) for probs in cnn_pred_probs])
        # Resolve activity labels
        cnn_activity_predictions = [activity_map[pred] for pred in cnn_predictions]
        
        # Add predictions to the DataFrame
        data['LogisticRegression_Prediction'] = lr_predictions
        data['LSTM_Prediction'] = lstm_activity_predictions
        data['CNN_Prediction'] = cnn_activity_predictions

        # Function to highlight mismatches
        def highlight_mismatch(val, actual):
            """Return color if mismatch, else no style."""
            return 'background-color: lightcoral' if val != actual else ''

        # Apply styling to mismatched predictions
        styled_df = data.style.applymap(lambda val: highlight_mismatch(val, data[actual_activity_col]),
                                        subset=['LogisticRegression_Prediction', 'LSTM_Prediction', 'CNN_Prediction'])

        # Display predictions
        st.write("🔍 **Predictions:**")
        #st.dataframe(data)
        st.dataframe(styled_df, use_container_width=True)


    except Exception as e:
        st.error(f"Error processing data: {e}")
