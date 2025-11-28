# 📡 Telecom Customer Churn Prediction

A machine learning web application that predicts whether a telecom customer is likely to churn based on their demographic and service usage details.  
The app is built using **Streamlit**, **Scikit-Learn**, **XGBoost**, **TensorFlow (ANN & CNN)**, and provides real-time predictions through a simple interactive UI.

---

## ✨ Features

- 🔮 **Predict churn probability** using:
  - XGBoost Model  
  - Artificial Neural Network (ANN)  
  - 1D Convolutional Neural Network (CNN)  
- ⚙️ **Automatic preprocessor loading** 
- 🧠 **Real-time predictions** with model selection  
- 📊 **User-friendly input form**  
- 🗂️ Uses multiple categorical & numerical telecom features  

---
## 🌐 Live Demo  
[🚀 Click Here to Open Live App](https://telecom-churn-prediction-786s.onrender.com/)


---

## 🧩 Input Features

The model uses the following customer attributes:

- Gender  
- Senior Citizen  
- Partner  
- Dependents  
- Phone Service  
- Multiple Lines  
- Internet Service  
- Online Security  
- Online Backup  
- Device Protection  
- Tech Support  
- Streaming TV  
- Streaming Movies  
- Contract  
- Paperless Billing  
- Payment Method  
- Tenure  
- Monthly Charges  
- Total Charges  

---




## 📂 Project Structure

```
telecom-churn-prediction/
│
├── telecomm_prediction.py
├── model_pipeline.joblib 
├── preprocessor.joblib
├── ann_model.h5 
├── cnn1d_model.h5 
│
├── WA_Fn-UseC_-Telco-Customer-Churn.csv 
└──  requirements.txt 

```

---

## 🚀 Getting Started

### 1️⃣ Clone the repository
```bash
git clone https://github.com/bhanuprasad1226/telecom-churn-prediction.git
cd telecom-churn-prediction
```
### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```
### 3️⃣ Run the application
```bash
streamlit run telecomm_prediction.py
```
- Open your browser at: `http://localhost:8501/`

---
## 🧠 Models Included
🔹 **XGBoost Classifier**

- Full pipeline saved using joblib

- Includes preprocessing + XGBClassifier

- Fast and accurate

🔹 **ANN (Keras / TensorFlow)**

- Fully connected layers

- Accepts preprocessed numeric input

- Outputs churn probability

🔹 **CNN-1D Model**

- 1D convolution over the numeric feature space

- Good for pattern extraction

---
 ## 🛠 Technologies Used

- Python 3.10+

- Streamlit for UI

- Scikit-Learn for preprocessing

- XGBoost for classification

- TensorFlow / Keras for ANN & CNN models

- Pandas / NumPy for data processing



