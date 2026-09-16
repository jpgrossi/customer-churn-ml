RANDOM_STATE = 42
TEST_SIZE = 0.20

ID_COL = 'customerID'
TARGET = 'Churn'

NUMERIC_FEATURES = ['tenure', 'MonthlyCharges', 'TotalCharges']

CATEGORICAL_FEATURES = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents',
    'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod'
]

MODEL_DIR = 'models'
MODEL_FILE = 'churn_pipeline.joblib'
