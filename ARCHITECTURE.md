# System Architecture Documentation

## Overview

The AI Manhwa Dubbing System is designed as a modular, scalable pipeline for automated video content creation. This document details the architectural design, component interactions, and data flow.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Input Layer                                  │
│  • Manhwa Chapter Images (JPG/PNG)                              │
│  • Configuration Files (YAML)                                    │
│  • API Credentials                                               │
└────────────────────┬────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────┐
│                 Processing Pipeline                              │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 1. OCR Module                                             │  │
│  │    • Image Preprocessing                                  │  │
│  │    • Text Region Detection                                │  │
│  │    • Character Recognition (Tesseract)                    │  │
│  │    • Text Extraction & Structuring                        │  │
│  └────────────────┬─────────────────────────────────────────┘  │
│                   │                                             │
│  ┌────────────────▼─────────────────────────────────────────┐  │
│  │ 2. AI Recap Generation                                    │  │
│  │    • Text Aggregation                                     │  │
│  │    • GPT-4 Summarization                                  │  │
│  │    • Narrative Structuring                                │  │
│  │    • Context Preservation                                 │  │
│  └────────────────┬─────────────────────────────────────────┘  │
│                   │                                             │
│  ┌────────────────▼─────────────────────────────────────────┐  │
│  │ 3. Panel Detection & Cropping                             │  │
│  │    • Panel Boundary Detection (YOLO/OpenCV)               │  │
│  │    • Panel Ordering & Sequencing                          │  │
│  │    • Image Cropping & Extraction                          │  │
│  │    • Quality Enhancement                                  │  │
│  └────────────────┬─────────────────────────────────────────┘  │
│                   │                                             │
│  ┌────────────────▼─────────────────────────────────────────┐  │
│  │ 4. Text-to-Speech Synthesis                               │  │
│  │    • Emotion Detection                                    │  │
│  │    • Voice Selection (Male/Female)                        │  │
│  │    • Audio Generation (ElevenLabs)                        │  │
│  │    • Prosody & Expression Control                         │  │
│  └────────────────┬─────────────────────────────────────────┘  │
│                   │                                             │
│  ┌────────────────▼─────────────────────────────────────────┐  │
│  │ 5. Video Assembly                                         │  │
│  │    • Panel Sequencing                                     │  │
│  │    • Audio-Video Synchronization                          │  │
│  │    • Background Music Integration                         │  │
│  │    • Effects & Transitions                                │  │
│  │    • Final Encoding (FFmpeg)                              │  │
│  └────────────────┬─────────────────────────────────────────┘  │
│                   │                                             │
└───────────────────┼─────────────────────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────────────────────┐
│                  Output Layer                                    │
│  • MP4 Video File (1080p, H.264)                                │
│  • Metadata JSON                                                 │
│  • Processing Logs                                               │
│  • (Optional) YouTube Upload                                     │
└──────────────────────────────────────────────────────────────────┘
```

## Component Architecture

### 1. OCR Module

**Purpose:** Extract all text content from manhwa images

**Components:**
- **ImagePreprocessor**: Enhances image quality for OCR
  - Grayscale conversion
  - Noise reduction
  - Adaptive thresholding
  - Contrast enhancement

- **TextRegionDetector**: Identifies text regions
  - Speech bubble detection
  - Text box identification
  - Region of interest extraction

- **TextExtractor**: Performs OCR
  - Tesseract OCR engine integration
  - Multi-language support
  - Confidence scoring
  - Post-processing corrections

**Data Flow:**
```
Raw Image → Preprocessing → Region Detection → OCR → Structured Text
```

**Key Technologies:**
- Tesseract OCR 4.x+
- OpenCV for image processing
- PIL/Pillow for image manipulation
- NumPy for numerical operations

**Input:** Manhwa page images (JPEG/PNG)
**Output:** Structured JSON with extracted text and locations

```json
{
  "page_id": "chapter_01_page_05",
  "text_regions": [
    {
      "id": 1,
      "bbox": [100, 150, 300, 200],
      "text": "This is the dialogue text",
      "confidence": 0.95,
      "type": "speech_bubble"
    }
  ]
}
```

### 2. AI Recap Generation Module

**Purpose:** Generate coherent narrative summaries from extracted text

**Components:**
- **TextAggregator**: Combines text from multiple pages/chapters
- **PromptEngineer**: Creates effective GPT-4 prompts
- **SummaryGenerator**: Interfaces with OpenAI API
- **NarrativeStructurer**: Organizes summary into coherent flow

**Summarization Strategy:**

1. **Chunking**: Break large text into manageable chunks (4K tokens each)
2. **Initial Summarization**: Summarize each chunk independently
3. **Meta-Summarization**: Combine chunk summaries into final recap
4. **Refinement**: Polish narrative flow and coherence

**Prompt Engineering:**
```python
RECAP_PROMPT = """
You are creating an engaging YouTube video recap of a manhwa series.
Based on the following text from chapters, create a compelling narrative summary that:

1. Captures key plot points and character development
2. Maintains chronological flow
3. Highlights dramatic moments
4. Uses engaging, YouTube-friendly language
5. Includes cliffhangers and hooks

Text to summarize:
{extracted_text}

Generate a {target_length} word recap suitable for voice-over narration.
"""
```

**API Integration:**
```python
import openai

def generate_recap(text: str, max_tokens: int = 8000) -> str:
    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are a skilled storyteller."},
            {"role": "user", "content": RECAP_PROMPT.format(extracted_text=text)}
        ],
        max_tokens=max_tokens,
        temperature=0.7
    )
    return response.choices[0].message.content
```

**Input:** Extracted text from all chapters
**Output:** Coherent narrative recap (5000-8000 words)

### 3. Panel Detection & Cropping Module

**Purpose:** Automatically detect and extract individual manhwa panels

**Components:**
- **PanelDetector**: Identifies panel boundaries
  - Contour detection
  - YOLO object detection (optional)
  - Boundary refinement

- **PanelOrderer**: Determines reading order
  - Top-to-bottom, left-to-right logic
  - Row detection
  - Sequence assignment

- **PanelCropper**: Extracts individual panels
  - Bounding box extraction
  - Margin handling
  - Resolution optimization

**Detection Methods:**

**Method 1: OpenCV Contour Detection**
```python
import cv2

def detect_panels_opencv(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, 
                                     cv2.CHAIN_APPROX_SIMPLE)
    
    panels = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 100 and h > 100:  # Filter small artifacts
            panels.append((x, y, w, h))
    
    return panels
```

**Method 2: Deep Learning (YOLO)**
```python
from ultralytics import YOLO

model = YOLO('panel_detection_model.pt')
results = model.predict('manhwa_page.jpg')
panels = results[0].boxes.xyxy  # Bounding boxes
```

**Panel Ordering Algorithm:**
```python
def order_panels(panels):
    # Sort by vertical position first (rows)
    rows = []
    current_row = []
    row_threshold = 50  # pixels
    
    sorted_panels = sorted(panels, key=lambda p: p[1])  # Sort by y
    
    for panel in sorted_panels:
        if not current_row or abs(panel[1] - current_row[0][1]) < row_threshold:
            current_row.append(panel)
        else:
            rows.append(sorted(current_row, key=lambda p: p[0]))  # Sort by x
            current_row = [panel]
    
    if current_row:
        rows.append(sorted(current_row, key=lambda p: p[0]))
    
    return [panel for row in rows for panel in row]
```

**Input:** Manhwa page images
**Output:** Ordered sequence of cropped panel images

### 4. Text-to-Speech Module

**Purpose:** Generate emotional, human-like voice narration

**Components:**
- **EmotionAnalyzer**: Detects emotions in text
- **VoiceSelector**: Chooses appropriate voice
- **TTSEngine**: Generates audio (ElevenLabs)
- **AudioProcessor**: Post-processing and normalization

**Emotion Detection:**
```python
from transformers import pipeline

emotion_classifier = pipeline("text-classification", 
                              model="j-hartmann/emotion-english-distilroberta-base")

def analyze_emotions(text):
    segments = split_into_segments(text, max_length=500)
    emotions = []
    
    for segment in segments:
        result = emotion_classifier(segment)[0]
        emotions.append({
            "text": segment,
            "emotion": result['label'],
            "score": result['score']
        })
    
    return emotions
```

**ElevenLabs Integration:**
```python
from elevenlabs import generate, Voice, VoiceSettings

def generate_narration(text, voice_name="Rachel", emotion="expressive"):
    audio = generate(
        text=text,
        voice=Voice(
            voice_id=get_voice_id(voice_name),
            settings=VoiceSettings(
                stability=0.75,
                similarity_boost=0.85,
                style=0.5,
                use_speaker_boost=True
            )
        ),
        model="eleven_multilingual_v2"
    )
    
    return audio
```

**Emotional Voice Mapping:**
```python
EMOTION_VOICE_MAP = {
    "dramatic": {"stability": 0.5, "style": 0.8},
    "sad": {"stability": 0.7, "style": 0.3},
    "happy": {"stability": 0.6, "style": 0.7},
    "angry": {"stability": 0.4, "style": 0.9},
    "neutral": {"stability": 0.75, "style": 0.5}
}
```

**Input:** Recap text with emotion markers
**Output:** MP3/WAV audio file with narration

### 5. Video Assembly Module

**Purpose:** Combine panels, audio, and music into final video

**Components:**
- **VideoComposer**: Main video creation engine
- **AudioSynchronizer**: Syncs audio with visuals
- **EffectsProcessor**: Adds transitions and effects
- **MusicIntegrator**: Mixes background music
- **VideoEncoder**: Final encoding and export

**Video Creation Pipeline:**

```python
from moviepy.editor import *
import numpy as np

def create_video(panels, narration_audio, background_music=None):
    # Calculate timing
    audio_clip = AudioFileClip(narration_audio)
    total_duration = audio_clip.duration
    panel_duration = total_duration / len(panels)
    
    # Create video clips from panels
    clips = []
    for i, panel_path in enumerate(panels):
        img_clip = ImageClip(panel_path, duration=panel_duration)
        
        # Add Ken Burns effect (zoom + pan)
        img_clip = img_clip.resize(lambda t: 1 + 0.02 * t)  # Slight zoom
        
        # Add fade transition
        img_clip = img_clip.crossfadein(0.5).crossfadeout(0.5)
        
        clips.append(img_clip)
    
    # Concatenate all clips
    video = concatenate_videoclips(clips, method="compose")
    
    # Add narration audio
    video = video.set_audio(audio_clip)
    
    # Mix background music
    if background_music:
        bg_music = AudioFileClip(background_music).volumex(0.3)
        bg_music = bg_music.audio_loop(duration=total_duration)
        
        final_audio = CompositeAudioClip([audio_clip, bg_music])
        video = video.set_audio(final_audio)
    
    return video
```

**Effects & Transitions:**

1. **Ken Burns Effect**: Slight zoom and pan for visual interest
2. **Crossfade**: Smooth transitions between panels
3. **Motion Graphics**: Optional text overlays
4. **Color Grading**: Optional color enhancement

**Audio Mixing:**
```python
def mix_audio(narration, background_music, voice_volume=1.0, music_volume=0.3):
    from pydub import AudioSegment
    
    # Load audio files
    voice = AudioSegment.from_file(narration)
    music = AudioSegment.from_file(background_music)
    
    # Adjust volumes
    voice = voice + (voice_volume * 20 - 20)  # Convert to dB
    music = music + (music_volume * 20 - 20)
    
    # Loop music to match voice duration
    if len(music) < len(voice):
        loops = len(voice) // len(music) + 1
        music = music * loops
    
    music = music[:len(voice)]
    
    # Overlay
    mixed = voice.overlay(music)
    
    return mixed
```

**Video Encoding:**
```python
def encode_video(video_clip, output_path, quality="high"):
    quality_settings = {
        "high": {
            "fps": 24,
            "codec": "libx264",
            "bitrate": "8000k",
            "preset": "medium"
        },
        "medium": {
            "fps": 24,
            "codec": "libx264",
            "bitrate": "4000k",
            "preset": "fast"
        }
    }
    
    settings = quality_settings[quality]
    
    video_clip.write_videofile(
        output_path,
        fps=settings["fps"],
        codec=settings["codec"],
        bitrate=settings["bitrate"],
        preset=settings["preset"],
        audio_codec="aac",
        audio_bitrate="192k"
    )
```

**Input:** Panel images, narration audio, background music
**Output:** Final MP4 video file

## Data Flow Diagram

```
┌─────────────┐
│ Manhwa      │
│ Images      │
└──────┬──────┘
       │
       ▼
┌─────────────┐     ┌──────────────┐
│ OCR         │────▶│ Text JSON    │
│ Extraction  │     └──────┬───────┘
└─────────────┘            │
                           ▼
                    ┌──────────────┐
                    │ AI Recap     │
                    │ Generation   │
                    └──────┬───────┘
                           │
       ┌───────────────────┴───────────────────┐
       │                                        │
       ▼                                        ▼
┌─────────────┐                         ┌──────────────┐
│ Panel       │                         │ TTS Audio    │
│ Detection   │                         │ Generation   │
└──────┬──────┘                         └──────┬───────┘
       │                                        │
       │                                        │
       └────────────────┬───────────────────────┘
                        │
                        ▼
                 ┌──────────────┐
                 │ Video        │
                 │ Assembly     │
                 └──────┬───────┘
                        │
                        ▼
                 ┌──────────────┐
                 │ Final MP4    │
                 │ Output       │
                 └──────────────┘
```

## Technology Stack Details

### Core Libraries

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| OCR | Tesseract | 4.x+ | Text recognition |
| Image Processing | OpenCV | 4.8+ | Panel detection, preprocessing |
| Video Editing | MoviePy | 1.0.3+ | Video assembly |
| Video Encoding | FFmpeg | 4.x+ | Final video encoding |
| Image Manipulation | Pillow | 10.x+ | Image operations |
| Numerical | NumPy | 1.24+ | Array operations |
| AI Summarization | OpenAI API | GPT-4 | Text summarization |
| Text-to-Speech | ElevenLabs API | - | Voice generation |
| Panel Detection | YOLOv8 (optional) | 8.x | Deep learning detection |

### API Dependencies

1. **OpenAI GPT-4 API**
   - Purpose: Text summarization
   - Cost: ~$0.03 per 1K tokens (input), $0.06 per 1K tokens (output)
   - Rate Limits: Varies by tier

2. **ElevenLabs API**
   - Purpose: Text-to-speech synthesis
   - Cost: Character-based pricing
   - Free tier: 10,000 characters/month
   - Paid: Starting at $5/month

3. **Background Music** (Optional)
   - Beatoven.ai, Soundraw, or royalty-free libraries
   - Cost: Varies or free

## Scalability Considerations

### Parallel Processing

```python
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing

def process_chapters_parallel(chapters, max_workers=None):
    if max_workers is None:
        max_workers = multiprocessing.cpu_count()
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for chapter in chapters:
            future = executor.submit(process_single_chapter, chapter)
            futures.append(future)
        
        results = [future.result() for future in futures]
    
    return results
```

### Caching Strategy

```python
import hashlib
import pickle
from functools import wraps

def cache_result(cache_dir="cache"):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            key = hashlib.md5(
                str((args, kwargs)).encode()
            ).hexdigest()
            
            cache_file = f"{cache_dir}/{func.__name__}_{key}.pkl"
            
            # Check cache
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            
            # Compute result
            result = func(*args, **kwargs)
            
            # Save to cache
            os.makedirs(cache_dir, exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
            
            return result
        return wrapper
    return decorator
```

### Resource Management

- **Memory**: Process chapters in batches to avoid memory overflow
- **Disk**: Use temporary directories, cleanup after processing
- **API**: Implement rate limiting and retry logic
- **GPU**: Optional GPU acceleration for panel detection

## Error Handling & Logging

```python
import logging
from functools import wraps

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def log_errors(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            logger.info(f"Starting {func.__name__}")
            result = func(*args, **kwargs)
            logger.info(f"Completed {func.__name__}")
            return result
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
            raise
    return wrapper
```

## Performance Optimization

### GPU Acceleration

```python
import torch

def use_gpu_if_available():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device

# For panel detection with YOLO
model = YOLO('model.pt')
model.to(use_gpu_if_available())
```

### Batch Processing

Process multiple pages simultaneously to reduce API overhead:

```python
def batch_ocr(image_paths, batch_size=10):
    results = []
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i+batch_size]
        batch_results = parallel_ocr(batch)
        results.extend(batch_results)
    return results
```

## Security Considerations

1. **API Key Management**: Store in environment variables, never commit
2. **Input Validation**: Validate all input files and parameters
3. **Sandboxing**: Run untrusted code in isolated environments
4. **Rate Limiting**: Implement API rate limiting to prevent abuse
5. **Data Privacy**: Handle user data according to privacy policies

---

**Next:** See [API_REFERENCE.md](API_REFERENCE.md) for detailed API documentation.
