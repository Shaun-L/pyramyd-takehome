FROM python:3.12

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

# Set environment variable to avoid Python buffering
ENV PYTHONUNBUFFERED=1

CMD ["python", "app.py"]
