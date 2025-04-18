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

# Define the model class (same as in train_model.py)
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim=468*3, num_classes=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.net(x.flatten(1))

# Load model
device = torch.device('cpu')  # Use CPU to avoid GPU overload
model = SimpleClassifier(num_classes=6).to(device)
model.load_state_dict(torch.load('expression_classifier.pt', map_location=device, weights_only=True))
model.eval()
scaler = load('scaler.pkl')
label_encoder = load('label_encoder.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        start_time = time.time()
        data = request.get_json()
        print(f"[{datetime.now()}] Received landmarks:", len(data['landmarks']))
        landmarks = np.array(data['landmarks'], dtype=np.float32)
        print(f"[{datetime.now()}] Landmarks shape:", landmarks.shape)
        if landmarks.shape[0] != 1404:
            return jsonify({'error': f'Expected 1404 features, got {landmarks.shape[0]}'}), 400
        landmarks = scaler.transform(landmarks.reshape(1, -1))
        print(f"[{datetime.now()}] Scaled landmarks shape:", landmarks.shape)
        with torch.no_grad():
            pred = model(torch.tensor(landmarks, dtype=torch.float32).to(device))
            print(f"[{datetime.now()}] Prediction output:", pred)
            pred_idx = torch.argmax(pred, dim=1).item()
            expression = label_encoder.inverse_transform([pred_idx])[0]
            print(f"[{datetime.now()}] Predicted expression:", expression)
        print(f"[{datetime.now()}] Prediction took {time.time() - start_time:.3f} seconds")
        return jsonify({'expression': expression})
    except Exception as e:
        print(f"[{datetime.now()}] Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)