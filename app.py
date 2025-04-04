import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Increase Pandas Styler render limit
pd.set_option("styler.render.max_elements", 500000)  # Adjust based on your dataset size



# Load the model
#model = pickle.load(open("Logistic_Regression.pkl", 'rb'))
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

       # ff_pred_probs = models['FeedForward'].predict(data)
       # feedforward_predictions = np.array([np.argmax(probs) for probs in ff_pred_probs])
        # Resolve activity labels
       # ff_activity_predictions = [activity_map[pred] for pred in feedforward_predictions]

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
       # data['FeedForward'] = feedforward_predictions
        data['LSTM_Prediction'] = lstm_activity_predictions
        data['CNN_Prediction'] = cnn_activity_predictions

        # Function to color mismatched predictions
        def highlight_mismatch(pred_col, actual_col):
            def highlight(row):
                return ['background-color: lightcoral' if row[pred_col] != row[actual_col] else '' for _ in row]
            return highlight
            
        # Apply highlight to each prediction column
        styled_df = data.style.apply(highlight_mismatch('LogisticRegression_Prediction', 'Activity'), axis=1, subset=['LogisticRegression_Prediction', 'Activity']) \
          #  .apply(highlight_mismatch('LSTM_Prediction', 'Activity'), axis=1, subset=['LSTM_Prediction']) \
          #  .apply(highlight_mismatch('CNN_Prediction', 'Activity'), axis=1, subset=['CNN_Prediction'])

        # Display predictions
        st.write("🔍 **Predictions:**")
        #st.dataframe(data)
        st.dataframe(styled_df, use_container_width=True)


    except Exception as e:
        st.error(f"Error processing data: {e}")
