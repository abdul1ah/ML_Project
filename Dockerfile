FROM python:3.9-bookworm

# 1. Create the user
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

# 2. Install essential build tools
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 3. Pre-install heavy dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir "numpy<2.0.0"

# 4. Install requirements
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of the application
COPY --chown=user . .

# 6. FIX: Ensure directories exist and permissions are set as Root
# We also create the Models folder in case the script needs to save there
RUN mkdir -p /home/user/app/.prefect /home/user/app/Data /home/user/app/Models && \
    chown -R user:user /home/user/app

# 7. Set up Environment and User
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=/home/user/app \
    PREFECT_HOME=/home/user/app/.prefect 

# 8. DEBUG STEP: List files to see if Data/movie_mapping.csv exists
RUN ls -R /home/user/app/Data

# 9. Run Training
RUN python workflow/train_pipeline.py

# 10. Final settings for Hugging Face
EXPOSE 7860
CMD ["uvicorn", "Script.fastapi.backend:app", "--host", "0.0.0.0", "--port", "7860"]