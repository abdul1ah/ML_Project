FROM python:3.9-bookworm

# 1. Create the user
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

# 2. Install essential build tools
# We need g++ for scikit-surprise's C++ components
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 3. Pre-install heavy dependencies
# Installing numpy first helps scikit-surprise find headers without crashing
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir "numpy<2.0.0"

# 4. Install requirements
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of the application
COPY --chown=user . .

# 6. Set up Environment and User
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=/home/user/app

# 7. Run Training
# This ensures matching .pkl files exist before the app starts
RUN python workflow/train_pipeline.py

# 8. Final settings for Hugging Face
EXPOSE 7860
CMD ["uvicorn", "Script.fastapi.backend:app", "--host", "0.0.0.0", "--port", "7860"]