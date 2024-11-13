# Use a Python base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the local code into the container
COPY . /app

# Install the required dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port FastAPI will use
EXPOSE 8003

# Command to run the FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8003"]
