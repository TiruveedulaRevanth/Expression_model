from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import numpy as np
from joblib import load
import time
from datetime import datetime

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "http://localhost:5173"}})

# Model class
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim=468*3, num_classes=6):
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
        return self.net(x.flatten(1))

# Load model and assets
device = torch.device('cpu')
try:
    model = SimpleClassifier(num_classes=6).to(device)
    model.load_state_dict(torch.load('expression_classifier.pt', map_location=device, weights_only=True))
    model.eval()
    scaler = load('scaler.pkl')
    label_encoder = load('label_encoder.pkl')
    print("Model and assets loaded successfully")
except Exception as e:
    print(f"Error loading model or assets: {e}")
    exit(1)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        start_time = time.time()
        data = request.get_json()
        if not data or 'landmarks' not in data:
            return jsonify({'error': 'No landmarks provided'}), 400
        
        landmarks = np.array(data['landmarks'], dtype=np.float32)
        print(f"[{datetime.now()}] Received landmarks: {len(data['landmarks'])}")
        print(f"[{datetime.now()}] Landmarks shape: {landmarks.shape}")
        print(f"[{datetime.now()}] Sample landmarks: {landmarks[:5]}")  # Log first few values

        if landmarks.shape[0] != 1404:
            return jsonify({'error': f'Expected 1404 features, got {landmarks.shape[0]}'}), 400
        if np.any(np.isnan(landmarks)) or np.any(np.isinf(landmarks)):
            return jsonify({'error': 'Landmarks contain NaN or Inf values'}), 400

        landmarks = scaler.transform(landmarks.reshape(1, -1))
        print(f"[{datetime.now()}] Scaled landmarks shape: {landmarks.shape}")

        with torch.no_grad():
            pred = model(torch.tensor(landmarks, dtype=torch.float32).to(device))
            probs = torch.softmax(pred, dim=1).cpu().numpy()[0]
            pred_idx = torch.argmax(pred, dim=1).item()
            expression = label_encoder.inverse_transform([pred_idx])[0]
            confidence = float(probs[pred_idx])
            print(f"[{datetime.now()}] Prediction output: {pred}")
            print(f"[{datetime.now()}] Predicted expression: {expression} (Confidence: {confidence:.2f})")

        print(f"[{datetime.now()}] Prediction took {time.time() - start_time:.3f} seconds")
        return jsonify({'expression': expression, 'confidence': confidence})
    except Exception as e:
        print(f"[{datetime.now()}] Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'Server is running'}), 200

if __name__ == '__main__':
    app.run(port=5000, debug=True)