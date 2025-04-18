# capstone_har_model
 Human Activity Recognition for Health Monitoring Using Wearable Devices

1.	Introduction to the Problem statement :  Use sensor data from wearable devices to track physical activity.Identify potential health concerns, such as excessive inactivity or irregular movement patterns.
   Develop a classification model to recognize human activities (e.g., walking, sitting, standing).
2.	Getting the requisite dataset - https://www.kaggle.com/datasets/uciml/human-activity-recognition-with smartphones 
3.	Import the requisite Libraries and packages
4.	Data Upload and Reading the data
a.	Upload Data on Google drive and mount on Google Colab
b.	Import and requisite libraries 
c.	Load dataset from CSV file format into Panda’s data frame 
d.	Check basic statistics o .head o .info o .shape
5.	Exploring the dataset – data pre-processing 
a.	Performing Exploratory Data Analysis (EDA)
b.	Getting insight into the data to understand the data features
c.	Removing duplicates data
6.	Performing Feature Engineering to creating features  for Modeling 
a.	Performing Threshold Variance: It is used to remove features with low variance
b.	Performing Pearson Correlation: To assess the linear relationship between two continuous variables. 
c.	Performing Analysis of Variance (ANOVA) test - to analyse the variation within each group and between the groups
7.	Model Building and Evaluation: Machine learning algorithms are applied to build predictive models. 
a.	Logistic Regression Model
b.	Feed Forward Neural Network
c.	LSTM
d.	CNN
e.	LSTM+CNN
8.	Models are then evaluated using metrics like accuracy, precision, recall, or F1 score to ensure optimal performance.
9.	Cross Validation is performed
10.	Model Deployment: Using Streamlit createUI to deploy the working model.


