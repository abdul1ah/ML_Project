FROM python:3.9-slim

# Create a non-root user 'user' (Required by HF)
RUN useradd -m -u 1000 user

# Set working directory inside the user's home
WORKDIR /home/user/app

# Install system dependencies (build-essential for scikit-surprise)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install as the 'user'
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of your files
COPY --chown=user . .

# Switch to the non-root user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# HF expects port 7860
EXPOSE 7860

# Point to your specific subfolder path on port 7860
CMD ["uvicorn", "Script.fastapi.backend:app", "--host", "0.0.0.0", "--port", "7860"]