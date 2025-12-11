from flask import Flask, render_template, jsonify, request
import subprocess
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os


app = Flask(__name__)

# Loading the model
MODEL_PATH = r"C:\Users\jayap\OneDrive\Desktop\mini project\sign-language-project\sign_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)


CLASSES = [chr(i) for i in range(65, 91)]  # ['A', 'B', ..., 'Z']
@app.route("/Practice")
def Practice():
    return render_template("Practice.html")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/static_prediction')
def static_prediction():
    return render_template('static_home.html')

@app.route('/dynamic_prediction')
def dynamic_prediction():
    return render_template('dynamic_home.html')

@app.route('/letters_to_signs')
def letters_to_signs():
    return render_template('letters_to_signs.html')

@app.route('/signs_to_letters')
def signs_to_letters():
    return render_template('signs_to_letters.html')

@app.route('/predict_sign', methods=['POST'])
def predict_sign():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'})

    try:
        
        img = Image.open(io.BytesIO(file.read()))
        img = img.convert('RGB')
        img = img.resize((224, 224))


        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predictinig
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions[0])
        confidence = round(np.max(predictions[0]) * 100, 2)

        return jsonify({
            'predicted_class': CLASSES[predicted_index],
            'confidence': confidence
        })
    except Exception as e:
        return jsonify({'error': str(e)})


import sys
import subprocess
import os

@app.route("/run_dynamic", methods=["POST"])
def run_dynamic():
    try:
        import sys, os, subprocess

        
        python_exec = sys.executable.replace("python.exe", "pythonw.exe")
        if not os.path.exists(python_exec):
            python_exec = sys.executable

        script_path = r"C:\Users\jayap\OneDrive\Desktop\mini project\sign-language-project\realtime_recognition.py"
        working_dir = os.path.dirname(script_path)

       
        subprocess.Popen(
            [python_exec, script_path],
            cwd=working_dir,
            stdout=subprocess.DEVNULL,  
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            creationflags=subprocess.DETACHED_PROCESS  
        )

        return jsonify({"message": "Launching Dynamic Prediction window..."})
    except Exception as e:
        return jsonify({"message": f"Error launching: {str(e)}"})



if __name__ == '__main__':
    app.run(debug=True)
