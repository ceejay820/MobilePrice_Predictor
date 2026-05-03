from django.shortcuts import render
import joblib
import os
import numpy as np

# Load the model and scaler
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'mobile_price_model.pkl')
scaler_path = os.path.join(BASE_DIR, 'scaler.pkl')

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

def index(request):
    prediction = None
    inputs = {}
    if request.method == 'POST':
        try:
            # Extract features from POST request
            input_keys = [
                'battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 
                'four_g', 'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 
                'pc', 'px_height', 'px_width', 'ram', 'sc_h', 
                'sc_w', 'talk_time', 'three_g', 'touch_screen', 'wifi'
            ]
            
            # Save inputs to pass back to template
            inputs = {key: request.POST.get(key) for key in input_keys}
            
            features = [float(inputs[key]) for key in input_keys]
            
            # Scale features
            features_scaled = scaler.transform([features])
            
            # Make prediction
            pred_class = model.predict(features_scaled)[0]
            
            # Map prediction to human readable string
            price_map = {
                0: "Low Cost",
                1: "Medium Cost",
                2: "High Cost",
                3: "Very High Cost"
            }
            prediction = price_map.get(pred_class, "Unknown")
            
        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render(request, 'predictor/index.html', {
        'prediction': prediction,
        'inputs': inputs
    })
