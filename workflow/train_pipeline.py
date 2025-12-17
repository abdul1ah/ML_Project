from prefect import flow, task
import subprocess
import os
import sys

# Define root_dir once at the top
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

@task(name="Collaborative Training", retries=1)
def run_collaborative():
    result = subprocess.run(
        [sys.executable, "Script/models/collaborative.py"], 
        capture_output=True, 
        text=True,
        cwd=ROOT_DIR  # Consistent
    )
    if result.returncode != 0:
        raise Exception(f"Collaborative training failed: {result.stderr}")
    return "Collaborative artifacts saved."

@task(name="Content-Based Training", retries=1)
def run_content():
    result = subprocess.run(
        [sys.executable, "Script/models/content_based.py"], 
        capture_output=True, 
        text=True,
        cwd=ROOT_DIR  # Add this!
    )
    if result.returncode != 0:
        raise Exception(f"Content training failed: {result.stderr}")
    return "Content artifacts saved."

@task(name="Hybrid Assembly")
def run_hybrid(collab_status, content_status):
    result = subprocess.run(
        [sys.executable, "Script/models/hybrid.py"], 
        capture_output=True, 
        text=True,
        cwd=ROOT_DIR  # Add this!
    )
    if result.returncode != 0:
        raise Exception(f"Hybrid assembly failed: {result.stderr}")
    return "Hybrid artifacts ready for Backend."

@flow(name="Movie Recommendation Training Pipeline")
def training_pipeline():
    collab = run_collaborative()
    content = run_content()
    run_hybrid(collab, content)

if __name__ == "__main__":
    training_pipeline()