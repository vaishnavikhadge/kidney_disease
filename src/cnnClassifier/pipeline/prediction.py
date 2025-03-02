import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename
        self.model = load_model(os.path.join("model", "model.h5"))  # Load model once

    def predict(self):
        try:
            # Load and preprocess image
            test_image = image.load_img(self.filename, target_size=(224, 224))
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis=0)

            # Predict
            result = np.argmax(self.model.predict(test_image), axis=1)

            # Interpret results
            prediction = "Tumor" if result[0] == 1 else "Normal"
            return {"prediction": prediction}

        except Exception as e:
            return {"error": str(e)}
