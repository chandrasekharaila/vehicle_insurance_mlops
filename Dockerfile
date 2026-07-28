# 1. Use an updated slim Debian base image (or python:3.10-slim)
FROM python:3.10-slim-bookworm

# 2. Prevent Python from buffering stdout/stderr and writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 3. Set the working directory
WORKDIR /app

# 4. Copy requirements first to leverage Docker layer caching
COPY requirements.txt /app/requirements.txt

# 5. Install dependencies without caching wheel files to reduce image size
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 6. Copy the application source code
COPY . /app

# 7. Expose the port FastAPI runs on (8080 or 5000)
EXPOSE 8080

# 8. Run FastAPI using Uvicorn (Production ASGI Server)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]