PYTHON = python3
APP_NAME = triton_model_server
IMAGE_NAME = $(APP_NAME):latest

environment:
	$(PYTHON) -m pip install -r requirements.txt

run:
	sudo docker run -p 8003:8003 $(IMAGE_NAME)

package:
	sudo docker build -t $(IMAGE_NAME) .

start:
	uvicorn app:app --port 8003