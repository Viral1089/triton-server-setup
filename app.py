import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput

app = FastAPI()

# Triton server URL
TRITON_SERVER_INSTANCES = [
    "tritonserver:8000",  # Instance 1
    "tritonserver1:8005",  # Instance 2  
]

current_server_index = 0

def get_triton_client():
    global current_server_index
    server_url = TRITON_SERVER_INSTANCES[current_server_index]
    print(server_url)
    current_server_index = (current_server_index + 1) % len(TRITON_SERVER_INSTANCES)  # Round robin logic
    return InferenceServerClient(url=server_url)



@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read image data from request
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes))

    # Resize image to 640x640 and convert to numpy array (CHW format)
    img_resized = img.resize((640, 640))
    img_np = np.array(img_resized).transpose(2, 0, 1)  # CHW format
    img_np = img_np.astype(np.float32)  # Ensure it's in float32

    img_np = np.expand_dims(img_np, axis=0)

    # Create Triton input
    inputs = [
        InferInput("images", img_np.shape, "FP32")
    ]
    inputs[0].set_data_from_numpy(img_np)


    # Create Triton output
    outputs = [
        InferRequestedOutput("output0")
    ]

    # Triton client instance    
    client = get_triton_client()

    # Perform inference
    try:
        response = client.infer(model_name="yolo11n", inputs=inputs, outputs=outputs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in inference: {str(e)}")

    # Get model outputs (parse YOLO detections)
    output_data = response.as_numpy("output0")
     
    predictions = []

    for i in range(output_data.shape[0]):
        pred = output_data[i]
        predictions.append({
            "class_id": int(pred[i][0]),
            "confidence": float(pred[i][1]),
            "bbox": [float(coord) for coord in pred[i][2:]],
        })

    return {"predictions": predictions}

@app.get("/")
async def root():
    return {"message": "Welcome to the YOLO object detection API using Triton!"}
