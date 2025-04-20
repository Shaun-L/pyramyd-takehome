FROM python:3.12

# Set working directory
WORKDIR /app

# Copy requirements (we’ll define this next)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your app
COPY . .

# Expose port Flask is running on
EXPOSE 5000

# Set environment variable to avoid Python buffering
ENV PYTHONUNBUFFERED=1

# Run your app
CMD ["python", "app.py"]
