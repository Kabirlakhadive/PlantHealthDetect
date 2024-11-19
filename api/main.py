from io import BytesIO
from PIL import Image
import numpy as np
from fastapi import FastAPI, UploadFile, File
import uvicorn
import tensorflow as tf
import tensorflow.keras as keras
from starlette.middleware.cors import CORSMiddleware
from tensorflow.keras.layers import TFSMLayer, Input
from tensorflow.keras.models import Model
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (or specify ["http://localhost:3000"] to restrict)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Load TFSMLayer
tfs = TFSMLayer('../models/2', call_endpoint='serving_default')

# Define a new model with TFSMLayer as a layer
input_layer = Input(shape=(256, 256, 3))  # Replace with the shape expected by TFSMLayer
output_layer = tfs(input_layer)

MODEL = Model(inputs=input_layer, outputs=output_layer)

CLASS_NAMES = ['Pepper bell Bacterial spot','Pepper bell healthy','Potato Early blight','Potato Late_blight','Potato healthy']


@app.get("/ping")
async def ping():
    return "Hello, I am alive"


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.post("/predict")
async def predict(
        file: UploadFile = File(...),
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image,0)
    predictions = MODEL.predict(img_batch)

    # Access the nested dictionary key to get the actual prediction values
    if isinstance(predictions, dict):
        prediction = predictions.get("output_0", [])[0]
    else:
        prediction = predictions[0]  # Fallback in case it’s not a dictionary

    # Get the index of the highest probability class
    predicted_class_index = np.argmax(prediction)

    # Get the class name and confidence
    predicted_class = CLASS_NAMES[predicted_class_index]
    confidence = prediction[predicted_class_index]

    return {
        "class": predicted_class, "confidence": float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)