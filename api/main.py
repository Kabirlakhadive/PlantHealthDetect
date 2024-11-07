from io import BytesIO
from PIL import Image
import numpy as np
from fastapi import FastAPI, UploadFile, File
import uvicorn
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import TFSMLayer, Input
from tensorflow.keras.models import Model
import os

app = FastAPI()

# Load TFSMLayer
tfs = TFSMLayer('../models/1', call_endpoint='serving_default')

# Define a new model with TFSMLayer as a layer
input_layer = Input(shape=(256, 256, 3))  # Replace with the shape expected by TFSMLayer
output_layer = tfs(input_layer)

MODEL = Model(inputs=input_layer, outputs=output_layer)

CLASS_NAMES = ['Pepper__bell___Bacterial_spot',
               'Pepper__bell___healthy',
               'Potato___Early_blight',
               'Potato___Late_blight',
               'Potato___healthy']


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
    img_batch = np.expand_dims(image, axis=0)
    predictions = MODEL.predict(img_batch)
    pass

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
