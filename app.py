from flask import render_template, jsonify, Flask, request
import os
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
from keras import backend as K
import requests
import time
import math

app = Flask(__name__)

# -----------------------------
# Skin disease classes
# -----------------------------
SKIN_CLASSES = {
    0: 'Actinic Keratoses (Solar Keratoses) / Bowen’s disease',
    1: 'Basal Cell Carcinoma',
    2: 'Benign Keratosis',
    3: 'Dermatofibroma',
    4: 'Melanoma',
    5: 'Melanocytic Nevi',
    6: 'Vascular skin lesion'
}

# -----------------------------
# Severity + Precautions + Food + Consultation info
# -----------------------------
DISEASE_INFO = {
    0: {
        "severity": "Moderate to High",
        "precautions": [
            "Use sunscreen SPF 30+ daily.",
            "Avoid direct sunlight.",
            "Regularly monitor for changes."
        ],
        "food_precautions": [
            "Eat fresh fruits and vegetables.",
            "Drink enough water.",
            "Avoid junk and oily food.",
            "Reduce sugar intake.",
            "Avoid alcohol and smoking."
        ],
        "consultation": "Dermatologist consultation strongly recommended."
    },

    1: {
        "severity": "High",
        "precautions": [
            "Avoid UV exposure.",
            "Do not scratch the lesion.",
            "Track lesion growth with photos."
        ],
        "food_precautions": [
            "Eat healthy home-cooked food.",
            "Include fruits and vegetables daily.",
            "Drink enough water.",
            "Avoid alcohol completely.",
            "Avoid processed and junk food."
        ],
        "consultation": "Immediate dermatologist visit recommended."
    },

    2: {
        "severity": "Low to Moderate",
        "precautions": [
            "Avoid skin irritation.",
            "Moisturize dry skin.",
            "Monitor for color or size changes."
        ],
        "food_precautions": [
            "Maintain a balanced diet.",
            "Drink enough water.",
            "Reduce oily and spicy food.",
            "Avoid excess sugar.",
            "Prefer home food."
        ],
        "consultation": "Routine dermatology checkup advised."
    },

    3: {
        "severity": "Low",
        "precautions": [
            "Avoid trauma to the lesion.",
            "Keep skin clean.",
            "Watch for sudden size or texture changes."
        ],
        "food_precautions": [
            "Eat nutritious food.",
            "Include proteins like pulses or eggs.",
            "Drink enough water.",
            "Avoid junk food.",
            "Avoid excessive caffeine."
        ],
        "consultation": "Non-urgent dermatology visit recommended."
    },

    4: {
        "severity": "Very High (dangerous)",
        "precautions": [
            "Avoid sun completely.",
            "Do not delay medical checkup.",
            "Monitor ABCDE(Asymmetry, Border irregularity, Color variation, Diameter(>6 mm), Evolving) signs."
        ],
        "food_precautions": [
            "Eat antioxidant-rich fruits and vegetables.",
            "Drink plenty of water.",
            "Avoid alcohol and smoking.",
            "Avoid processed food.",
            "Maintain a healthy diet to support treatment."
        ],
        "consultation": "Urgent dermatologist or oncologist appointment needed."
    },

    5: {
        "severity": "Low (but monitor)",
        "precautions": [
            "Monitor mole changes.",
            "Use sunscreen daily.",
            "Avoid self-removal."
        ],
        "food_precautions": [
            "Eat fresh fruits and vegetables.",
            "Drink enough water.",
            "Avoid junk food.",
            "Reduce sugar intake.",
            "Maintain a healthy lifestyle."
        ],
        "consultation": "Consult dermatologist if changes appear."
    },

    6: {
        "severity": "Low to Moderate",
        "precautions": [
            "Avoid scratching.",
            "Keep the area clean.",
            "Watch for sudden swelling or bleeding."
        ],
        "food_precautions": [
            "Drink enough water.",
            "Eat balanced meals.",
            "Reduce salty food.",
            "Avoid alcohol.",
            "Avoid junk food."
        ],
        "consultation": "Dermatology consultation recommended."
    }
}


# -----------------------------
# Haversine Distance Calculation
# -----------------------------
def calculate_distance(lat1, lon1, lat2, lon2):
    try:
        R = 6371
        lat1 = math.radians(float(lat1))
        lon1 = math.radians(float(lon1))
        lat2 = math.radians(float(lat2))
        lon2 = math.radians(float(lon2))

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        return round(R * c, 2)
    except:
        return None


# -----------------------------
# Home Page
# -----------------------------
@app.route('/')
def index():
    return render_template('index.html', title='Home')


# -----------------------------
# Prediction Upload Handler
# -----------------------------
@app.route('/uploaded', methods=['POST'])
def uploaded():
    f = request.files['file']
    path = 'static/data/' + f.filename
    f.save(path)

    # Load complete Keras model directly
    model = load_model("modelnew.h5", compile=False)

    # Preprocess Image
    img = image.load_img(path, target_size=(224, 224))
    img = np.array(img).reshape(1, 224, 224, 3) / 255.0

    # Predict Disease
    prediction = model.predict(img)
    pred_class = np.argmax(prediction)

    disease = SKIN_CLASSES[pred_class]
    accuracy = float(prediction[0][pred_class]) * 100
    info = DISEASE_INFO[pred_class]

    K.clear_session()

    return render_template(
        'uploaded.html',
        img_file=f.filename,
        predictions=disease,
        acc=round(accuracy, 2),
        severity_level=info["severity"],
        precautions=info["precautions"],
        food_precautions=info["food_precautions"], 
        consultation=info["consultation"]
    )


# -----------------------------
# Nearby Hospitals API
# Using OSM + Default Location
# -----------------------------
@app.route('/nearby_hospitals')
def nearby_hospitals():

    DEFAULT_LAT = "16.3067"
    DEFAULT_LNG = "80.4365"

    # Get GPS location from frontend
    lat = request.args.get("lat")
    lng = request.args.get("lng")

    if not lat or not lng:
        lat, lng = DEFAULT_LAT, DEFAULT_LNG

    # Bounding Box (local area)
    VIEWBOX = "80.20,16.50,80.70,16.10"

    try:
        time.sleep(1)  # OSM usage rule

        url = (
            "https://nominatim.openstreetmap.org/search?"
            "q=hospital&format=json&limit=5&addressdetails=1"
            f"&bounded=1&viewbox={VIEWBOX}"
        )

        headers = {"User-Agent": "SkinDiseaseDetector/1.0"}
        results = requests.get(url, headers=headers).json()

        hospitals = []

        for r in results:
            dist = calculate_distance(lat, lng, r["lat"], r["lon"])

            hospitals.append({
                "name": r.get("display_name", "Unknown Hospital"),
                "address": r.get("address", {}).get("road", "Not Provided"),
                "lat": r["lat"],
                "lng": r["lon"],
                "distance": dist
            })

        return jsonify({
            "status": "ok",
            "latitude_used": lat,
            "longitude_used": lng,
            "hospitals": hospitals
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


# -----------------------------
# Run App
# -----------------------------
if __name__ == '__main__':
    app.run(debug=False)