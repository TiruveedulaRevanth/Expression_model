import json

# Generate 1404 values (0.1 repeated)
landmarks = [0.1] * (468 * 3)

# Save to JSON file
with open("landmarks.json", "w") as f:
    json.dump({"landmarks": landmarks}, f)

print("Generated landmarks.json with 1404 values")