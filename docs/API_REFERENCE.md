# ⚙️ API Reference — AI Manhwa Dubbing System

## Base URL
```
http://localhost:8000/api/v1
```

## Endpoints

### POST /dubbing/start
Starts a dubbing process.
```json
{
  "project_name": "solo_leveling",
  "language": "en",
  "voice": "female_expressive",
  "chapters": [1, 2, 3]
}
```
**Response**
```json
{
  "task_id": "b3d91e12",
  "status": "started",
  "estimated_time": "25m"
}
```

### GET /status/{task_id}
Retrieves job status.
```json
{
  "task_id": "b3d91e12",
  "status": "processing",
  "progress": 72,
  "current_stage": "tts_generation"
}
```

### GET /result/{task_id}
Returns final video URLs and logs.
```json
{
  "video_url": "https://cdn.manhwa.ai/output/solo_leveling.mp4",
  "logs": "https://cdn.manhwa.ai/logs/b3d91e12.log"
}
```

### GET /health
Health check endpoint.
```json
{"status": "ok"}
```
