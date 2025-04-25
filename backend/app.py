from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import numpy as np
from joblib import load
import os
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "http://localhost:5173"}})

MODEL_PATH = os.path.join("model", "expression_classifier.pt")
SCALER_PATH = os.path.join("model", "scaler.pkl")
ENCODER_PATH = os.path.join("model", "label_encoder.pkl")

# Model class
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim=1404, num_classes=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# Load model and assets
device = torch.device('cpu')
model = SimpleClassifier().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))  # Added weights_only=True
model.eval()

scaler = load(SCALER_PATH)
label_encoder = load(ENCODER_PATH)

@app.route('/predict', methods=['POST'])
def predict():
    # Log request details
    logger.debug(f"Request headers: {request.headers}")
    logger.debug(f"Content-Type: {request.content_type}")

    # Validate Content-Type
    if request.content_type != 'application/json':
        logger.error("Invalid Content-Type")
        return jsonify({'error': 'Content-Type must be application/json'}), 415

    try:
        data = request.get_json()
        logger.debug(f"Received data: {data.keys()}")

        landmarks = data.get("landmarks")
        if not landmarks or len(landmarks) != 1404:
            logger.error(f"Invalid landmarks length: {len(landmarks) if landmarks else 'None'}")
            return jsonify({'error': 'Expected 1404 landmark values'}), 400

        landmarks_np = np.array(landmarks, dtype=np.float32).reshape(1, -1)

        if np.any(np.isnan(landmarks_np)) or np.any(np.isinf(landmarks_np)):
            logger.error("Invalid values in landmarks")
            return jsonify({'error': 'Invalid values in landmarks'}), 400

        scaled = scaler.transform(landmarks_np)
        tensor = torch.tensor(scaled, dtype=torch.float32).to(device)

        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))
            expression = label_encoder.inverse_transform([pred_idx])[0]
            confidence = round(float(probs[pred_idx]), 4)

        logger.info(f"Prediction: {expression} ({confidence})")
        return jsonify({'expression': expression, 'confidence': confidence})

    except Exception as e:
        logger.exception(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'OK'})

if __name__ == '__main__':
    app.run(port=5000, debug=True, host='0.0.0.0')  # Added host='0.0.0.0'