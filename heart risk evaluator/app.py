from flask import Flask, render_template, request
import joblib  
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

model = joblib.load('lg_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def form():
    return render_template('form.html')

@app.route('/submit', methods=['POST'])
def submit():
    # Retrieve form data
    orig_data = request.form
    data = [
        request.form.get('age'),
        request.form.get('gender'),
        request.form.get('cholesterol'),
        request.form.get('blood_pressure'),
        request.form.get('heart_rate'),
        request.form.get('exercise_hours'),
        request.form.get('family_history'),
        request.form.get('diabetes'),
        request.form.get('obesity'),
        request.form.get('blood_sugar'),
    ]

    chest_pain = request.form.get('chest_pain')
    if chest_pain == "0":
        data.append("1")
        data.append("0")
        data.append("0")
    elif chest_pain == "1":
        data.append("0")
        data.append("1")
        data.append("0")
    elif chest_pain == "2":
        data.append("0")
        data.append("0")
        data.append("1")

    data = [float(string) for string in data]
    input_data = np.array(data).reshape(1, -1)
    scaled_data = scaler.transform(input_data)
    # Make prediction using the ML model
    prediction = model.predict(scaled_data)

    print(prediction)

    return render_template('result.html', data=data, prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
