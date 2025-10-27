# ⚙️ Configuration Guide — AI Manhwa Dubbing System

## config.yaml Structure

```yaml
paths:
  input: ./data/input
  output: ./data/output
  cache: ./data/cache

services:
  translator: deepl
  tts: elevenlabs
  voice_clone: openvoice

pipeline:
  concurrency: 4
  logging: detailed

api:
  port: 8000
  auth: false
```
**Notes:**
- `pipeline.concurrency` controls number of chapters processed in parallel.
- `tts` supports local models (Coqui, VITS).
- Environment variables override config.yaml values.
