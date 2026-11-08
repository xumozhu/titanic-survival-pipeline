
# Use Python 3.11 as the base runtime environment.
FROM python:3.11-slim

# Set the working directory inside the Docker container.
WORKDIR /app

# Copy the dependency list into the container.
COPY requirements.txt .

# Install the required Python packages without caching downloaded files.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Python script into the container.
COPY src/python/titanic_model.py src/python/titanic_model.py

# Create the directory where Titanic CSV files will be mounted.
RUN mkdir -p src/data

# Run the Titanic prediction script when the container starts.
CMD ["python", "src/python/titanic_model.py"]
