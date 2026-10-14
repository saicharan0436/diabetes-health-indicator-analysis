import requests

url = "http://127.0.0.1:5000/predict"

data = {
    "HighBP": 1,
    "HighChol": 0,
    "BMI": 27,
    "Smoker": 0,
    "Stroke": 0,
    "HeartDiseaseorAttack": 0,
    "PhysActivity": 1,
    "Fruits": 1,
    "Veggies": 1,
    "HvyAlcoholConsump": 0,
    "GenHlth": 2,
    "MentHlth": 1,
    "PhysHlth": 2,
    "DiffWalk": 0,
    "Sex": 1,
    "Age": 7,
    "Education": 5,
    "Income": 6
}

response = requests.post(url, json=data)

print("Status Code:", response.status_code)
print("Response:", response.json())