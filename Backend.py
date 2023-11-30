# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 16:21:49 2023

@author: Pratesh Mishra
"""

import numpy as np
from flask import Flask, render_template, request
from pymongo import MongoClient
import pickle
from Model_crop_recommendation import scaler

app = Flask(__name__)
client = MongoClient("mongodb://localhost:27017")
db = client["Credential_CR_user"]
collection = db["Credential_signup"]

model = pickle.load(open(r"C:\major project\Model_ensumble.pkl", "rb"))


@app.route('/')
def index():
    return render_template('home.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form['username']
        password = request.form['password']

        # Check if the email is not already in the database
        if collection.find_one({'email': email}):
            return "Email already exists. Please choose a different email."

        # Insert user data into the MongoDB collection
        user_data = {'email': email, 'password': password}
        collection.insert_one(user_data)

        return render_template("home_log.html") 

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['username']
        password = request.form['password']

        # Check if the user with the given email and password exists in the database
        user = collection.find_one({'email': email, 'password': password})

        if user:
            return render_template("home_log.html")
        else:
            return "Invalid email or password. Please try again."

    return render_template('login.html')

@app.route('/home_log', methods=['GET', 'POST'])
def home_log():

    return render_template('home_log.html')

@app.route('/home', methods=['GET', 'POST'])
def logout():
    return render_template('home.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        nitrogen = float(request.form.get('Nitrogen'))
        phosphorous = float(request.form.get('Phosphorous'))
        potassium = float(request.form.get('Potassium'))
        temperature = float(request.form.get('Temperature'))
        humidity = float(request.form.get('Humidity'))
        ph = float(request.form.get('pH'))
        rainfall = float(request.form.get('Rainfall'))

        # Store the values in an array
        data = [np.array([nitrogen, phosphorous, potassium, temperature, humidity, ph, rainfall])]
        scaled_user_input = scaler.transform(data)
        input_data = np.array(scaled_user_input).reshape(1, 7)
        prediction = model.predict(input_data)


        return f"Recommended Crop: {prediction}"

    return render_template('predict.html')




if __name__ == '__main__':
    app.run(debug=True)