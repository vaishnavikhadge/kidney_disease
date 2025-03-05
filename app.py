from flask import Flask, request, jsonify, render_template
import os
import uuid
import base64
import subprocess
from flask_cors import CORS, cross_origin
from cnnClassifier.pipeline.prediction import PredictionPipeline

os.environ["LANG"] = "en_US.UTF-8"
os.environ["LC_ALL"] = "en_US.UTF-8"

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

@app.route("/train", methods=['POST'])
@cross_origin()
def trainRoute():
    try:
        result = subprocess.run(["python", "main.py"], check=True, capture_output=True, text=True)
        return jsonify({"message": "Training done successfully!", "output": result.stdout}), 200
    except subprocess.CalledProcessError as e:
        return jsonify({"error": "Training failed", "details": e.stderr}), 500

@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    try:
        if 'image' in request.files:  # If image is sent as a file
            image_file = request.files['image']
            if image_file.filename == '':
                return jsonify({'error': 'No selected file'}), 400

            # Generate a unique filename and save
            filename = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}.jpg")
            image_file.save(filename)

        elif request.is_json and 'image' in request.get_json():  # If image is sent as Base64
            data = request.get_json()
            image_data = base64.b64decode(data['image'])
            filename = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}.jpg")
            with open(filename, "wb") as f:
                f.write(image_data)

        else:
            return jsonify({'error': 'No image provided'}), 400

        # Run Prediction
        classifier = PredictionPipeline(filename)
        result = classifier.predict()

        return jsonify(result), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8080)
