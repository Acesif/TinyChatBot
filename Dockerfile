# Use the Python 3.9 base image
FROM python:3.10

# Create a non-root user
RUN useradd -m -u 1000 user
USER user

# Set up the environment path
ENV PATH="/home/user/.local/bin:$PATH"

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the application code
COPY --chown=user . /app

# Command to run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
