from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import json
import io
from PIL import Image
import os
import requests

# ✅ Create FastAPI instance
app = FastAPI()

# ✅ Optional: allow frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_URL = "https://drive.google.com/uc?export=download&id=1bM4hQm502mIQ9vfJaWr6aHRJ5Grn3V5T"
MODEL_PATH = "model.h5"

# Check if model exists
if not os.path.exists(MODEL_PATH):
    print("Downloading model.h5...")
    with requests.get(MODEL_URL, stream=True) as r:
        r.raise_for_status()
        with open(MODEL_PATH, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print("✅ model.h5 downloaded!")

# ✅ Load model
model = load_model("model.h5")

# ✅ Map class index to name
class_map = {
    0: "Bacterial Spot",
    1: "Early Blight",
    2: "Healthy",
    3: "Late Blight",
    4: "Leaf Mold",
    5: "Septoria Leaf Spot",
    6: "Target Spot",
    7: "Yellow Leaf Curl Virus",
    8: "Mosaic Virus",
    9: "Two-Spotted Spider Mite"
}

# ✅ Load extra disease info
with open("disease_info.json", "r") as file:
    disease_details = json.load(file)

# ✅ Image preprocessing
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((128, 128))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# ✅ Predict route
@app.post("/predict_leaf")
async def predict_leaf(file: UploadFile = File(...)):
    contents = await file.read()
    image = preprocess_image(contents)
    prediction = model.predict(image)
    class_index = np.argmax(prediction)
    class_name = class_map[class_index]
    confidence = round(prediction[0][class_index] * 100, 2)

    details = disease_details.get(class_name, {
        "scientific_name": "Unknown",
        "recommended_treatment": "Not available",
        "symptoms": "Not available"
    })

    return {
    "plant": details.get("plant", "Tomato"),
    "prediction": class_name,
    "confidence": f"{confidence}%",
    "common_name": details.get("common_name", "Not available"),
    "scientific_name": details.get("scientific_name", "Not available"),
    "recommended_treatment": details.get("recommended_treatment", "Not available"),
    "symptoms": details.get("symptoms", "Not available")
}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)