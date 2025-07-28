from flask import Flask, render_template, request
import pickle
import numpy as np
import joblib 
import os 

app = Flask(__name__)

# -----------------------
# Load the model and encoders
# -----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, 'model.pkl'))


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data = joblib.load(os.path.join(BASE_DIR, 'encoders.pkl'))
scaler = data["scaler"]
label_encoders = data["label_encoders"]
output_encoders = data["output_encoders"]

# -----------------------
# Prediction Function
# -----------------------
def predict_diet(age, gender, weight, height, bmi, disease_type, severity,
                 activity, calories, cholesterol, blood_pressure, glucose, exercise):
    # Prepare input data
    data = np.array([[age, gender, weight, height, bmi, disease_type,
                      severity, activity, calories, cholesterol,
                      blood_pressure, glucose, exercise]])

    # Encode categorical features
    data[0, 1] = label_encoders["Gender"].transform([gender])[0]
    data[0, 5] = label_encoders["Disease_Type"].transform([disease_type])[0]
    data[0, 6] = label_encoders["Severity"].transform([severity])[0]
    data[0, 7] = label_encoders["Physical_Activity_Level"].transform([activity])[0]

    # Scale numeric features
    numeric_indices = [0, 2, 3, 4, 8, 9, 10, 11, 12]
    data[:, numeric_indices] = scaler.transform(data[:, numeric_indices])

    # Predict
    prediction = model.predict(data)[0]

    # Decode predictions
    decoded_pred = {}
    i = 0
    for key in output_encoders.keys():
        decoded_pred[key] = output_encoders[key].inverse_transform([prediction[i]])[0]
        i += 1

    return decoded_pred

# -----------------------
# Routes
# -----------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        age = int(request.form['age'])
        gender = request.form['gender']
        weight = float(request.form['weight'])
        height = float(request.form['height'])
        bmi = float(request.form['bmi'])
        disease_type = request.form['disease_type']
        severity = request.form['severity']
        activity = request.form['activity']
        calories = float(request.form['calories'])
        cholesterol = float(request.form['cholesterol'])
        blood_pressure = float(request.form['blood_pressure'])
        glucose = float(request.form['glucose'])
        exercise = float(request.form['exercise'])

        prediction = predict_diet(age, gender, weight, height, bmi, disease_type,
                                  severity, activity, calories, cholesterol,
                                  blood_pressure, glucose, exercise)
    return render_template('index.html', prediction=prediction)

# -----------------------
# Run Flask App
# -----------------------
if __name__ == '__main__':
    app.run(debug=True)
