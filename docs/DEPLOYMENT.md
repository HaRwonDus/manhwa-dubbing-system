# 🚀 Deployment Guide — AI Manhwa Dubbing System

## Local Development
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn backend.main:app --reload
```

## Docker Deployment
```bash
docker-compose up --build -d
```
Example `docker-compose.yml`:
```yaml
version: '3.8'
services:
  backend:
    build: ./backend
    ports: ["8000:8000"]
    volumes:
      - ./data:/app/data
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ELEVENLABS_API_KEY=${ELEVENLABS_API_KEY}
  frontend:
    build: ./frontend
    ports: ["3000:3000"]
```
