import json
import numpy as np
landmarks = np.random.uniform(0, 1, 1404).tolist()  # Random values for testing
with open("landmarks.json", "w") as f:
    json.dump({"landmarks": landmarks}, f)