from ultralytics import YOLO
from pathlib import Path
import onnx

# Load the YOLO model (change to your desired YOLO version)
model = YOLO('yolo11n.pt')

onnx_file = model.export(format="onnx", dynamic = True)

# Define paths
model_name = "yolo11n"
triton_repo_path = Path("models")
triton_model_path = triton_repo_path / model_name

# Create directories
(triton_model_path / "1").mkdir(parents=True, exist_ok=True)

Path(onnx_file).rename(triton_model_path / "1" / "model.onnx")

# Create config file
(triton_model_path / "config.pbtxt").touch()

# (Optional) Enable TensorRT for GPU inference
# First run will be slow due to TensorRT engine conversion
data = """
name: "yolo11n"
platform: "onnxruntime_onnx"
max_batch_size: 4
default_model_filename: "model.onnx"
input [
    {
        name: "images"
        data_type: TYPE_FP32
        dims: [3, 640, 640]
    }
]
output [
    {
        name: "output0"
        data_type: TYPE_FP32
        dims: [84,-1]
    }
]
instance_group [
    {
        kind: KIND_CPU
        count: 5
    }
]
"""

with open(triton_model_path / "config.pbtxt", "w") as f:
    f.write(data)

model = onnx.load('./models/yolo11n/1/model.onnx')

# You can downgrade the model to IR version 9 using onnx API
model.ir_version = 9
onnx.save(model, './models/yolo11n/1/model.onnx')

