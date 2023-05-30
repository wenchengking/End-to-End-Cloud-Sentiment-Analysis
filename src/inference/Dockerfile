# Use the official Python image as the base image and ensure that we target x86_64 architectures
FROM --platform=linux/x86_64 python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Update pip to use the latest version
RUN /usr/local/bin/python -m pip install --upgrade pip

# Copy the requirements file to the working directory
COPY requirements.txt .

# Install the required dependencies
RUN pip install -r requirements.txt

# Copy the Streamlit application files to the working directory
COPY src .

# Expose port 80 for http traffic
EXPOSE 80

# Set the command to run the Streamlit application
CMD ["streamlit", "run", "--server.port=80", "--server.fileWatcherType=none", "app.py"]
