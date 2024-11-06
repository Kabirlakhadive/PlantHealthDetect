from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import TFSMLayer
from tensorflow.keras.layers import Input

app = FastAPI()

# origins = [
#     "http://localhost",
#     "http://localhost:3000",
# ]
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# Load the model using TFSMLayer
tfs_layer = TFSMLayer('e:/plant project try 2/saved_models/6', call_endpoint='serving_default')

# Define a new model with TFSMLayer as a layer
input_layer = Input(shape=(256, 256, 3))  # Replace with the shape expected by TFSMLayer
output_layer = tfs_layer(input_layer)


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
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)