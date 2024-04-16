from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
import pandas as pd
import pickle

app = Flask(__name__)

# Load heart disease prediction model and related files
heart_disease_model = pickle.load(open('heart_disease_model.sav', 'rb'))
heart_disease_encoders = pickle.load(open('heart_disease_encoders.sav', 'rb'))
heart_disease_scaler = pickle.load(open('heart_disease_scaler.sav', 'rb'))

# Load diabetes prediction model and scaler
diabetes_model = pickle.load(open('diabetes_model.sav', 'rb'))
diabetes_scaler = pickle.load(open('diabetes_model_scaler.sav', 'rb'))

# Load pneumonia prediction model
pneumonia_model = load_model("pneumonia_prediction_model.h5")

# Define image preprocessing function for pneumonia prediction
def preprocess_image(image, img_size):
    try:
        img_arr = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        resized_arr = cv2.resize(img_arr, (img_size, img_size))
        normalized_img = resized_arr / 255.0
        processed_img = normalized_img.reshape(-1, img_size, img_size, 1)
        return processed_img
    except Exception as e:
        print(e)
        return None

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/heart_disease', methods=['GET', 'POST'])
def heart_disease():
    if request.method == 'POST':
        # Get form data
        data = {}
        data['age'] = float(request.form['age'])
        data['sex'] = request.form['sex']
        data['cp'] = request.form['cp']
        data['trestbps'] = float(request.form['trestbps'])
        data['chol'] = float(request.form['chol'])
        data['fbs'] = request.form['fbs']
        data['restecg'] = request.form['restecg']
        data['thalch'] = float(request.form['thalch'])
        data['exang'] = request.form['exang']
        data['oldpeak'] = float(request.form['oldpeak'])
        data['slope'] = request.form['slope']

        # Preprocess form data
        if data['fbs'] =='Yes':
            data['fbs'] = 1.0
        else:
            data['fbs'] = 0.0

        if data['exang'] == 'Yes':
            data['exang'] = 1.0
        else:
            data['exang'] = 0.0

        df = pd.DataFrame([data])

        for i in heart_disease_encoders['sex'].categories_[0]:
            df['sex' + '_' + i] = 0.0
        df['sex' + '_' + df['sex']] = 1.0
        df.drop(columns='sex', inplace=True)

        for i in heart_disease_encoders['cp'].categories_[0]:
            df['cp' + '_' + i] = 0.0
        df['cp' + '_' + df['cp']] = 1.0
        df.drop(columns='cp', inplace=True)

        for i in heart_disease_encoders['restecg'].categories_[0]:
            df['restecg' + '_' + i] = 0.0
        df['restecg' + '_' + df['restecg']] = 1.0
        df.drop(columns='restecg', inplace=True)

        for i in heart_disease_encoders['slope'].categories_[0]:
            df['slope' + '_' + i] = 0.0
        df['slope' + '_' + df['slope']] = 1.0
        df.drop(columns='slope', inplace=True)

        df = pd.DataFrame(heart_disease_scaler.transform(df), columns=df.columns)

        # Make prediction
        pred = heart_disease_model.predict(df)
        if pred == 1:
            pred = 'Heart disease, Please visit your Dr'
        else:
            pred = 'No Heart disease'
        return render_template('heart_disease.html', prediction=pred)
    else:
        return render_template('heart_disease.html')

@app.route('/diabetes', methods=['GET', 'POST'])
def diabetes():
    if request.method == 'POST':
        # Get form data
        data = {}
        data['Pregnancies'] = int(request.form.get('Pregnancies'))
        data['Glucose'] = int(request.form.get('Glucose'))
        data['BloodPressure'] = int(request.form.get('BloodPressure'))  
        data['SkinThickness'] = int(request.form.get('SkinThickness')) 
        data['Insulin'] = int(request.form.get('Insulin'))
        data['BMI'] = float(request.form.get('BMI'))
        data['DiabetesPedigreeFunction'] = float(request.form.get('DiabetesPedigreeFunction'))
        data['Age'] = int(request.form.get('Age'))

        df = pd.DataFrame([data])
        df = pd.DataFrame(diabetes_scaler.transform(df), columns=df.columns)

        # Make prediction
        pred = diabetes_model.predict(df)[0]
        if pred == 1:
            pred = 'Diabetes, Please visit your Dr'
        else:
            pred = 'Not Diabetes'
        return render_template('diabetes.html', prediction=pred)
    else:
        return render_template('diabetes.html')

@app.route('/pneumonia', methods=['GET', 'POST'])
def pneumonia():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            # Preprocess the uploaded image
            processed_image = preprocess_image(file, img_size=128)
            if processed_image is not None:
                # Make prediction using the loaded model
                prediction = pneumonia_model.predict(processed_image)[0]
                result = "Pneumonia" if prediction <= 0.5 else "Normal Lung"
                return render_template('pneumonia.html', prediction=result)
            else:
                return "Failed to process image"
        else:
            msg = "No file uploaded"
            return render_template('pneumonia.html', prediction=msg)
    else:
        return render_template('pneumonia.html')

if __name__ == '__main__':
    app.run(debug=True)
