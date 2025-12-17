FROM python:3.9-slim

# Create the user first
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install requirements
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all files and set ownership to 'user'
COPY --chown=user . .

# Switch to the non-root user BEFORE training
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Now train the model (the .pkl files will be owned by 'user')
RUN python workflow/train_pipeline.py

# Final settings
EXPOSE 7860
CMD ["uvicorn", "Script.fastapi.backend:app", "--host", "0.0.0.0", "--port", "7860"]