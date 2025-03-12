from flask import Flask, request, jsonify, render_template,send_file
from reportlab.pdfgen import canvas
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

@app.route("/result", methods=['GET'])
@cross_origin()
def result():
    return render_template("result.html")

@app.route("/train", methods=['POST'])
@cross_origin()
def trainRoute():
    try:
        result = subprocess.run(["python", "main.py"], check=True, capture_output=True, text=True)
        return jsonify({"message": "Training done successfully!", "output": result.stdout}), 200
    except subprocess.CalledProcessError as e:
        return jsonify({"error": "Training failed", "details": e.stderr}), 500

@app.route("/download/pdf", methods=['GET'])
def download_pdf():
    try:
        prediction = "Tumor Detected"  
        confidence = "92%"  
        
        pdf_filename = "prediction_report.pdf"
        pdf_path = os.path.join(UPLOAD_FOLDER, pdf_filename)
        
        c = canvas.Canvas(pdf_path)
        c.drawString(100, 750, "Kidney Disease Classification Report")
        c.drawString(100, 700, f"Prediction: {prediction}")
        c.drawString(100, 650, f"Confidence Score: {confidence}")
        c.save()
        
        return send_file(pdf_path, as_attachment=True)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

import csv

@app.route("/download/csv", methods=['GET'])
def download_csv():
    try:
        csv_filename = "prediction_results.csv"
        csv_path = os.path.join(UPLOAD_FOLDER, csv_filename)

        with open(csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Image Name", "Prediction", "Confidence"])
            writer.writerow(["sample.jpg", "Tumor Detected", "92%"])  # Example

        return send_file(csv_path, as_attachment=True)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


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
