# Use Python 3.10-slim as the base image
FROM python:3.10-slim
LABEL authors="Harishankar"

# Set the working directory in the container
WORKDIR /app

# Install dependencies
COPY requirements.django.txt /app/
RUN pip install -r requirements.django.txt

# Copy the current directory contents into the container
COPY . /app/

# Expose the port the app runs on
EXPOSE 8000

# Run the Django app
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]