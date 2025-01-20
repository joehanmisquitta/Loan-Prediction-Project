from imblearn.over_sampling import SMOTE
import streamlit as st
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.impute import SimpleImputer

# Reading the data
data = pd.read_csv("LoanApprovalPrediction.csv")

# Dropping Loan_ID column
data.drop(['Loan_ID'], axis=1, inplace=True)

# Creating label encoder object
label_encoder = preprocessing.LabelEncoder()

# Finding categorical columns
obj = (data.dtypes == 'object')
object_cols = list(obj[obj].index)

# Encoding categorical columns and storing mappings
encoding_maps = {}
for col in object_cols:
    data[col] = label_encoder.fit_transform(data[col])
    encoding_maps[col] = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

# Imputing missing values
imputer = SimpleImputer(strategy='mean')
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Splitting the data into train and test sets
X = data_imputed.drop(['Loan_Status'], axis=1)
Y = data_imputed['Loan_Status']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=1)

# Applying SMOTE to the training data
smote = SMOTE(random_state=1)
X_resampled, Y_resampled = smote.fit_resample(X_train, Y_train)

# Training the RandomForest model with the resampled data
rfc = RandomForestClassifier(n_estimators=7, criterion='entropy', random_state=7)
rfc.fit(X_resampled, Y_resampled)

# Predicting on the test set and evaluating
Y_pred = rfc.predict(X_test)
accuracy = 100 * metrics.accuracy_score(Y_test, Y_pred)
print("Accuracy: ", accuracy)

# Streamlit app
st.title("Loan Approval Prediction")

st.header("Data Overview")
st.write(X_resampled.head(15))

st.header("Encoding Mappings")
for col, encoding_map in encoding_maps.items():
    st.write(f'**Encoding for {col}:**')
    for original, encoded in encoding_map.items():
        st.write(f'{original}: {encoded}')
    st.write('')

st.header("Loan Approval Prediction")
gender = st.selectbox("Gender", options=["Male", "Female"])
married = st.selectbox("Married", options=["Yes", "No"])
dependents = st.selectbox("Dependents", options=["0", "1", "2", "3+"])
education = st.selectbox("Education", options=["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", options=["Yes", "No"])
applicant_income = st.number_input("Applicant Income", value=0)
coapplicant_income = st.number_input("Coapplicant Income", value=0)
loan_amount = st.number_input("Loan Amount", value=0)
loan_amount_term = st.number_input("Loan Amount Term", value=360)
credit_history = st.selectbox("Credit History", options=[0.0, 1.0])
property_area = st.selectbox("Property Area", options=["Urban", "Semiurban", "Rural"])

# Create input DataFrame for prediction
user_input = pd.DataFrame({
    'Gender': [gender],
    'Married': [married],
    'Dependents': [dependents],
    'Education': [education],
    'Self_Employed': [self_employed],
    'ApplicantIncome': [applicant_income],
    'CoapplicantIncome': [coapplicant_income],
    'LoanAmount': [loan_amount],
    'Loan_Amount_Term': [loan_amount_term],
    'Credit_History': [credit_history],
    'Property_Area': [property_area]
})

# Encode user input using the same mappings
for col in user_input.columns:
    if col in encoding_maps:
        user_input[col] = user_input[col].map(encoding_maps[col])

# Handle any potential NaN values after mapping
user_input.fillna(-1, inplace=True)

# Predict
if st.button("Predict"):
    prediction = rfc.predict(user_input)[0]
    status = "Congratulations! Your loan is Approved. 🎉" if prediction == 1 else "Sorry, your loan is Not Approved. 😔"
    st.write(f"Loan Status: {status}")













