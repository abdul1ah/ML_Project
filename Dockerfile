FROM python:3.9-slim

# 1. Create the user
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

# 2. Lighten the system dependencies
# We swap 'build-essential' for specific tools to avoid the Trixie-security bottleneck
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 3. Install requirements
# Copy only requirements first to leverage Docker cache
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 4. Copy the rest of the application
COPY --chown=user . .

# 5. Set up Environment and User
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=/home/user/app

# 6. Run Training
# This generates the .pkl files inside the container
RUN python workflow/train_pipeline.py

# 7. Final settings for Hugging Face
EXPOSE 7860
CMD ["uvicorn", "Script.fastapi.backend:app", "--host", "0.0.0.0", "--port", "7860"]