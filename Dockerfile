FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Let Cloud Run manage the port dynamically
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app