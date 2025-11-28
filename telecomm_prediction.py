

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
import streamlit as st
from tensorflow.keras import layers, models, callbacks, Input
from tensorflow.keras.models import load_model

# ----------------------------
# Config
# ----------------------------
MODEL_PATH = "model_pipeline.joblib"
PREPROC_PATH = "preprocessor.joblib"
ANN_PATH = "ann_model.h5"
CNN_PATH = "cnn1d_model.h5"
DATA_FILE = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

CATEGORICAL_COLS = [
    'gender','SeniorCitizen','Partner','Dependents','PhoneService','MultipleLines',
    'InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport',
    'StreamingTV','StreamingMovies','Contract','PaperlessBilling','PaymentMethod'
]
NUMERIC_COLS = ['tenure','MonthlyCharges','TotalCharges']
TARGET = 'Churn'

# ----------------------------
# Helpers
# ----------------------------

def build_preprocessor(categorical_cols=CATEGORICAL_COLS, numeric_cols=NUMERIC_COLS):
    try:
        cat_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    except TypeError:
        cat_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    num_pipe = Pipeline(steps=[('scaler', StandardScaler())])
    preproc = ColumnTransformer(transformers=[
        ('num', num_pipe, numeric_cols),
        ('cat', cat_encoder, categorical_cols)
    ])
    return preproc


def build_xgb_pipeline(preprocessor):
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=200, random_state=42)
    pipeline = Pipeline(steps=[('preproc', preprocessor), ('model', model)])
    return pipeline

# ----------------------------
# Loaders (automatic)
# ----------------------------
@st.cache_resource
def load_xgb_pipeline():
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            return None
    return None

@st.cache_resource
def load_preprocessor():
    # Prefer explicit preprocessor file
    if os.path.exists(PREPROC_PATH):
        try:
            return joblib.load(PREPROC_PATH)
        except Exception:
            return None
    # Fallback: extract preprocessor from saved XGB pipeline if present
    if os.path.exists(MODEL_PATH):
        try:
            pipe = joblib.load(MODEL_PATH)
            if isinstance(pipe, Pipeline):
                p = pipe.named_steps.get('preproc') or pipe.named_steps.get('preprocessor')
                if p is not None:
                    try:
                        joblib.dump(p, PREPROC_PATH)
                    except Exception:
                        pass
                    return p
        except Exception:
            return None
    return None

@st.cache_resource
def load_ann():
    if os.path.exists(ANN_PATH):
        try:
            return load_model(ANN_PATH)
        except Exception:
            return None
    return None

@st.cache_resource
def load_cnn():
    if os.path.exists(CNN_PATH):
        try:
            return load_model(CNN_PATH)
        except Exception:
            return None
    return None

# ----------------------------
# Minimal Streamlit UI
# ----------------------------
st.set_page_config(page_title="Telecom Churn Predictor", layout='centered')
st.title("Telecom Customer Churn - Predictor")
st.write("Select a model, enter customer features, and click Predict.")

# Auto-load models
xgb_pipeline = load_xgb_pipeline()
preprocessor = load_preprocessor()
ann_model = load_ann()
cnn_model = load_cnn()

# Sidebar: only model choice
model_choice = st.sidebar.selectbox("Model for prediction", ["XGBoost", "ANN", "CNN-1D"]) 

# Informational notes (concise)
if model_choice == 'XGBoost' and xgb_pipeline is None:
    st.sidebar.error("XGBoost pipeline not found. Please provide model_pipeline.joblib if you want XGBoost.")
if model_choice == 'ANN' and ann_model is None:
    st.sidebar.error("ANN model not found. Please provide ann_model.h5.")
if model_choice == 'CNN-1D' and cnn_model is None:
    st.sidebar.error("CNN model not found. Please provide cnn1d_model.h5.")

# Input form
with st.form('input_form'):
    st.subheader('Customer features')
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox('Gender', options=['Male','Female'])
        senior = st.selectbox('SeniorCitizen', options=['0','1'])
        partner = st.selectbox('Partner', options=['Yes','No'])
        dependents = st.selectbox('Dependents', options=['Yes','No'])
        tenure = st.number_input('tenure (months)', min_value=0, max_value=1000, value=12)
        monthly = st.number_input('MonthlyCharges', min_value=0.0, max_value=10000.0, value=70.0)
    with col2:
        phone = st.selectbox('PhoneService', options=['Yes','No'])
        multiple = st.selectbox('MultipleLines', options=['Yes','No','No phone service'])
        internet = st.selectbox('InternetService', options=['DSL','Fiber optic','No'])
        online_sec = st.selectbox('OnlineSecurity', options=['Yes','No','No internet service'])
        online_back = st.selectbox('OnlineBackup', options=['Yes','No','No internet service'])
        device = st.selectbox('DeviceProtection', options=['Yes','No','No internet service'])
    streaming_tv = st.selectbox('StreamingTV', options=['Yes','No','No internet service'])
    streaming_movies = st.selectbox('StreamingMovies', options=['Yes','No','No internet service'])
    contract = st.selectbox('Contract', options=['Month-to-month','One year','Two year'])
    paperless = st.selectbox('PaperlessBilling', options=['Yes','No'])
    payment = st.selectbox('PaymentMethod', options=['Electronic check','Mailed check','Bank transfer (automatic)','Credit card (automatic)'])
    if internet == 'No':
        tech = st.selectbox('TechSupport', options=['No internet service'])
    else:
        tech = st.selectbox('TechSupport', options=['Yes','No'])
    submit = st.form_submit_button('Predict')

if submit:
    input_df = pd.DataFrame([{
        'tenure': tenure,
        'MonthlyCharges': monthly,
        'TotalCharges': monthly * tenure if tenure>0 else monthly,
        'gender': gender,
        'SeniorCitizen': senior,
        'Partner': partner,
        'Dependents': dependents,
        'PhoneService': phone,
        'MultipleLines': multiple,
        'InternetService': internet,
        'OnlineSecurity': online_sec,
        'OnlineBackup': online_back,
        'DeviceProtection': device,
        'TechSupport': tech,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless,
        'PaymentMethod': payment
    }])

    if model_choice == 'XGBoost':
        if xgb_pipeline is None:
            st.error('XGBoost pipeline not available.')
        else:
            try:
                proba = float(xgb_pipeline.predict_proba(input_df)[0,1])
                st.write('**Model:** XGBoost')
                st.write('**Churn probability:**', f'{proba:.3f}')
                st.write('**Predicted churn:**', 'Yes' if proba>0.5 else 'No')
            except Exception as e:
                st.error(f'Prediction failed: {e}')

    elif model_choice == 'ANN':
        if preprocessor is None:
            st.error('Preprocessor not available for ANN. Provide preprocessor.joblib or model_pipeline.joblib.')
        elif ann_model is None:
            st.error('ANN model not available.')
        else:
            try:
                X_in = preprocessor.transform(input_df)
                if hasattr(X_in, 'toarray'):
                    X_in = X_in.toarray()
                X_in = np.asarray(X_in, dtype=np.float32)
                proba = float(ann_model.predict(X_in).ravel()[0])
                st.write('**Model:** ANN')
                st.write('**Churn probability:**', f'{proba:.3f}')
                st.write('**Predicted churn:**', 'Yes' if proba>0.5 else 'No')
            except Exception as e:
                st.error(f'ANN prediction failed: {e}')

    else:  # CNN-1D
        if preprocessor is None:
            st.error('Preprocessor not available for CNN. Provide preprocessor.joblib or model_pipeline.joblib.')
        elif cnn_model is None:
            st.error('CNN model not available.')
        else:
            try:
                X_in = preprocessor.transform(input_df)
                if hasattr(X_in, 'toarray'):
                    X_in = X_in.toarray()
                X_in = np.asarray(X_in, dtype=np.float32)
                Xc = X_in.reshape((X_in.shape[0], X_in.shape[1], 1))
                proba = float(cnn_model.predict(Xc).ravel()[0])
                st.write('**Model:** CNN-1D')
                st.write('**Churn probability:**', f'{proba:.3f}')
                st.write('**Predicted churn:**', 'Yes' if proba>0.5 else 'No')
            except Exception as e:
                st.error(f'CNN prediction failed: {e}')



