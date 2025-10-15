# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Update system packages and install dependencies
# - build-essential: For compiling Python packages
# - ffmpeg: For audio/video processing
# - libsndfile1: For audio libraries like librosa/soundfile
# - libportaudio2: For the sounddevice package
# - libgl1-mesa-glx: A common dependency for opencv-python
RUN apt-get update && apt-get install -y --no-install-recommends\
    build-essential\ 
    ffmpeg\ 
    libsndfile1\ 
    libportaudio2\ 
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# use .env
COPY .env .

# Copy only the requirements file to leverage Docker cache
COPY requirements.txt

RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Copy the rest of the application code
COPY . .

# Make port 8003 available to the world outside this container
EXPOSE 8003

# Making working directory backend directory
WORKDIR /app/backend

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8003"]

