import requests
import numpy as np

# Dummy landmarks (468 * 3 = 1404 random values)
landmarks = np.random.rand(468 * 3).tolist()

# Send POST request
response = requests.post(
    "http://localhost:5000/predict",
    json={"landmarks": landmarks},
    headers={"Content-Type": "application/json"}
)

print(response.status_code)
print(response.json())