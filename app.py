from operator import index
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

import numpy as np

import math


app = Flask(__name__)
CORS(app)



def predictMDD (data):
    input_data = np.array([[float(data['fasoil']), float(data['pi']), float(data['mddr'])]])
    scaler_X = joblib.load('data/model1/scaler_X.pkl')
    # scaler_y = joblib.load('data/model1/scaler_y.pkl')
    input_data = scaler_X.fit_transform(input_data)
    model1 = load_model('data/model1/model_1.h5')
    prediction = model1.predict(input_data)
    prediction = np.array(prediction).reshape(-1, 1)
    # scaled_prediction = scaler_y.inverse_transform(prediction)
    return prediction[0][0]


def predictOMC (data):
    input_data = np.array([[float(data['fasoil']), float(data['pi']), float(data['mddr'])]])
    scaler_X = joblib.load('data/model2/scaler_X.pkl')
    scaler_y = joblib.load('data/model2/scaler_y.pkl')
    input_data = scaler_X.fit_transform(input_data)
    model1 = load_model('data/model2/model_2.h5')
    prediction = model1.predict(input_data)
    prediction = np.array(prediction).reshape(-1, 1)
    scaled_prediction = scaler_y.inverse_transform(prediction)
    return scaled_prediction[0][0]

def predictUCS (data):
    input_data = np.array([[float(data['fasoil']), float(data['pi']), float(data['mddr'])]])
    scaler_X = joblib.load('data/model3/scaler_X.pkl')
    scaler_y = joblib.load('data/model3/scaler_y.pkl')
    input_data = scaler_X.fit_transform(input_data)
    model1 = load_model('data/model3/model_3.h5')
    prediction = model1.predict(input_data)
    prediction = np.array(prediction).reshape(-1, 1)
    scaled_prediction = scaler_y.inverse_transform(prediction)
    return scaled_prediction[0][0]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predictMDD', methods=['GET','POST'])
def predictMDD1():
    if request.method == 'POST':
        data1 = request.get_json(force=True)
        print(data1)
        # predictionMDD = predictOMC(data)
        data = pd.read_excel("data/MDD.xlsx")

        fasoil_value = float(data1['fasoil'])
        second_col_value = float(data1['pi'])
        third_col_value = float(data1['mddr'])  # Replace 'third_col_key' with the actual key for the third column value in data1
        matched = 0
        target = fasoil_value
        rows = data
        if fasoil_value == 0:
            fourth_column_value = third_col_value
            matched = 4
        elif not data[np.round(data.iloc[:, 0],2) == fasoil_value].empty:     
            rows = data[np.round(data.iloc[:, 0],2) == fasoil_value]
            matched = 2
            target = third_col_value
            if len(rows) > 1 and not rows[np.round(rows.iloc[:, 2],3) == third_col_value].empty:
                rows = rows[np.round(rows.iloc[:, 2],3) == third_col_value]
                matched = 1
                target = second_col_value
                if len(rows) > 1 and not rows[np.round(rows.iloc[:, 1],3) == second_col_value]:
                    rows = rows[np.round(rows.iloc[:, 1],3) == second_col_value]
                    fourth_column_value = rows.iloc[0, 3]
                    matched = 4
        if matched != 4:
                lower = rows[rows.iloc[:, matched] < target].iloc[:, matched].max()
                higher = rows[rows.iloc[:, matched] > target].iloc[:, matched].min()

                # Get the corresponding rows
                lower_row = rows[np.round(rows.iloc[:, matched],3) == lower]
                higher_row = rows[np.round(rows.iloc[:, matched],3) == higher]

                if not lower_row.empty and not higher_row.empty:
                    lower_value = lower_row.iloc[0, 3]
                    higher_value = higher_row.iloc[0, 3]
                    proportion = (target - lower) / (higher - lower)
                    fourth_column_value = lower_value + proportion * (higher_value - lower_value)

                elif lower_row.empty:
                    fourth_column_value = higher_row.iloc[0, 3]
                elif higher_row.empty:
                    fourth_column_value = lower_row.iloc[0, 3]

        return jsonify({"predicted": str(fourth_column_value)})
    else :
        return render_template('model1.html')
    
@app.route('/predictOMC', methods=['GET','POST'])
def predictOMC1():
    if request.method == 'POST':
        data1 = request.get_json(force=True)
        print(data1)
        # predictionMDD = (data)
        data = pd.read_excel("data/omc.xlsx")

        fasoil_value = float(data1['fasoil'])
        second_col_value = float(data1['pi'])
        third_col_value = float(data1['mddr'])  # Replace 'third_col_key' with the actual key for the third column value in data1
        matched = 0
        target = fasoil_value
        rows = data
        if fasoil_value == 0:
            fourth_column_value = third_col_value
            matched = 4
        elif not data[np.round(data.iloc[:, 0],2) == fasoil_value].empty:     
            rows = data[np.round(data.iloc[:, 0],2) == fasoil_value]
            matched = 2
            target = third_col_value
            if len(rows) > 1 and not rows[np.round(rows.iloc[:, 2],3) == third_col_value].empty:
                rows = rows[np.round(rows.iloc[:, 2],3) == third_col_value]
                matched = 1
                target = second_col_value
                if len(rows) > 1 and not rows[np.round(rows.iloc[:, 1],3) == second_col_value]:
                    rows = rows[np.round(rows.iloc[:, 1],3) == second_col_value]
                    fourth_column_value = rows.iloc[0, 3]
                    matched = 4
        if matched != 4:
                lower = rows[rows.iloc[:, matched] < target].iloc[:, matched].max()
                higher = rows[rows.iloc[:, matched] > target].iloc[:, matched].min()

                # Get the corresponding rows
                lower_row = rows[np.round(rows.iloc[:, matched],3) == lower]
                higher_row = rows[np.round(rows.iloc[:, matched],3) == higher]

                if not lower_row.empty and not higher_row.empty:
                    lower_value = lower_row.iloc[0, 3]
                    higher_value = higher_row.iloc[0, 3]
                    proportion = (target - lower) / (higher - lower)
                    fourth_column_value = lower_value + proportion * (higher_value - lower_value)

                elif lower_row.empty:
                    fourth_column_value = higher_row.iloc[0, 3]
                elif higher_row.empty:
                    fourth_column_value = lower_row.iloc[0, 3]

        return jsonify({"predicted": str(fourth_column_value)})
    else :
        return render_template('model2.html')

@app.route('/predictUCS', methods=['GET','POST'])
def predictUCS1():
    if request.method == 'POST':
        data1 = request.get_json(force=True)
        print(data1)
        # predictionMDD = predictUCS(data)
        data = pd.read_excel("data/Model UCS.xlsx")

        fasoil_value = float(data1['fasoil'])
        second_col_value = float(data1['pi'])
        third_col_value = float(data1['mddr'])  # Replace 'third_col_key' with the actual key for the third column value in data1
        matched = 0
        target = fasoil_value
        rows = data
        if fasoil_value == 0:
            fourth_column_value = third_col_value
            matched = 4
        elif not data[np.round(data.iloc[:, 0],2) == fasoil_value].empty:     
            rows = data[np.round(data.iloc[:, 0],2) == fasoil_value]
            matched = 2
            target = third_col_value
            if len(rows) > 1 and not rows[np.round(rows.iloc[:, 2],3) == third_col_value].empty:
                rows = rows[np.round(rows.iloc[:, 2],3) == third_col_value]
                matched = 1
                target = second_col_value
                if len(rows) > 1 and not rows[np.round(rows.iloc[:, 1],3) == second_col_value]:
                    rows = rows[np.round(rows.iloc[:, 1],3) == second_col_value]
                    fourth_column_value = rows.iloc[0, 3]
                    matched = 4
        if matched != 4:
                print(matched)
                print(rows)
                lower = rows[rows.iloc[:, matched] < target].iloc[:, matched].max()
                higher = rows[rows.iloc[:, matched] > target].iloc[:, matched].min()

                # Get the corresponding rows
                lower_row = rows[rows.iloc[:, matched] == lower]
                higher_row = rows[rows.iloc[:, matched] == higher]
                print(higher_row.empty)
                print(lower_row.empty)
                if not lower_row.empty and not higher_row.empty:
                    lower_value = lower_row.iloc[0, 3]
                    higher_value = higher_row.iloc[0, 3]
                    proportion = (target - lower) / (higher - lower)
                    fourth_column_value = lower_value + proportion * (higher_value - lower_value)
                elif lower_row.empty and not higher_row.empty:
                    fourth_column_value = higher_row.iloc[0, 3]
                elif higher_row.empty and not lower_row.empty:
                    fourth_column_value = lower_row.iloc[0, 3]

                

        return jsonify({"predicted": str(fourth_column_value)})
    else :
        return render_template('model3.html') 

if __name__ == '__main__':
    app.run(debug=True, port=5002 )