FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1

# Install system deps needed for some Python packages (opencv, pillow, build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency list
COPY requirements-cpu.txt .

RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements-cpu.txt \
    && pip install --no-cache-dir "torch==2.9.1+cpu" -f https://download.pytorch.org/whl/cpu/torch_stable.html

# Copy app
COPY . .

EXPOSE 8000

CMD ["gunicorn", "--workers", "3", "--bind", "0.0.0.0:8000", "server:app"]
