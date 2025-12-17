---
title: Cinephile Hub
emoji: 🎬
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
app_port: 7860
image: abdullah2223312/movie-recommendation:latest
---

# 🎬 Cinephile Hub

A Movie Recommendation System powered by **FastAPI**, **Scikit-Surprise**, and **Prefect**.

## 🚀 Deployment Info
This project is automatically deployed via GitHub Actions.
- **Docker Image:** `abdullah2223312/movie-recommendation:latest`
- **Backend:** FastAPI
- **CI/CD:** GitHub Actions to Docker Hub & Hugging Face Spaces

## 🛠️ Local Setup
If you want to run this locally using the Docker image:

```bash
docker pull abdullah2223312/movie-recommendation:latest
docker run -p 7860:7860 abdullah2223312/movie-recommendation