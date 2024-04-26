from flask import Flask, request, jsonify, render_template
from waitress import serve
import pickle
import numpy as np
import torch
import torch.nn as nn
from logreg import LogisticRegressionModel
from randomforest import RandomForestModel
import os

app = Flask(__name__)

# Load the model
model = pickle.load(open('app/model_lr.pkl', 'rb'))

case_status_mapping = {
    "Laboratory-confirmed": [1, 0],
    "Probable": [0, 1]
}

sex_group_mapping = {
    "female" : [1, 0, 0],
    "male": [0, 0, 1],
    "other": [0, 0, 1]
}
age_group_mapping = {
    "0-9": [1, 0, 0, 0, 0, 0, 0, 0, 0],
    "10-19": [0, 1, 0, 0, 0, 0, 0, 0, 0],
    "20-29": [0, 0, 1, 0, 0, 0, 0, 0, 0],
    "30-39": [0, 0, 0, 1, 0, 0, 0, 0, 0],
    "40-49": [0, 0, 0, 0, 1, 0, 0, 0, 0],
    "50-59": [0, 0, 0, 0, 0, 1, 0, 0, 0],
    "60-69": [0, 0, 0, 0, 0, 0, 1, 0, 0],
    "70-79": [0, 0, 0, 0, 0, 0, 0, 1, 0],
    "80": [0, 0, 0, 0, 0, 0, 0, 0, 1]
}
ethnicity_group_mapping = {
    "American Indian/Alaska Native, Non-Hispanic": [1, 0, 0, 0, 0, 0, 0],
    "Asian, Non-Hispanic": [0, 1, 0, 0, 0, 0, 0],
    "Black, Non-Hispanic": [0, 0, 1, 0, 0, 0, 0],
    "Multiple/Other, Non-Hispanic": [0, 0, 0, 1, 0, 0, 0],
    "Native Hawaiian/Other Pacific Islander, Non-Hispanic": [0, 0, 0, 0, 1, 0, 0],
    "White, Non-Hispanic": [0, 0, 0, 0, 0, 1, 0],
    "Hispanic/Latino": [0, 0, 0, 0, 0, 0, 1]
}

hospital_mapping = {
    "yes": [1],
    "no": [0]
}

icu_mapping = {
    "yes": [1],
    "no": [0]
}

medical_mapping = {
    "yes": [1],
    "no": [0]
}

binary_mapping = {
    "yes": [1],
    "no": [0]
}
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # model = request.form['model']
    model_name = request.form['model']
    model = pickle.load(open(f'app/model_{model_name}.pkl', 'rb'))
    case_status = case_status_mapping[request.form['current_status']]
    sex_group = sex_group_mapping[request.form['sex']]
    age_group = age_group_mapping[request.form['age_group']]
    ethnicity_group = ethnicity_group_mapping[request.form['ethnicity']]
    hospital_yn = binary_mapping[request.form['hospital_yn']]
    icu_yn = binary_mapping[request.form['icu_yn']]
    medcond_yn = binary_mapping[request.form['medcond_yn']]

    input_features = np.array(case_status + sex_group + age_group + ethnicity_group + hospital_yn + icu_yn + medcond_yn)
    input_tensor = torch.FloatTensor(input_features).unsqueeze(0)
    with torch.no_grad():  # Ensure no gradients are computed during inference
        if(model_name == 'lr'):
            prediction = model(input_tensor)
            output = prediction.numpy()[0][0]
        else:
            prediction = model.predict(torch.FloatTensor(input_features))
            output = prediction

    
    return render_template('index.html', prediction_text='Probability of event: {:.4f}'.format(output))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000)) 
    serve(app, host="0.0.0.0", port=port)