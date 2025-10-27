# 🧠 System Architecture Documentation  
### AI Manhwa Dubbing System

## 1. Overview

The **AI Manhwa Dubbing System** is a modular and scalable pipeline that automates the creation of high-quality manhwa recap videos for YouTube.  
It performs **OCR text extraction**, **AI summarization**, **panel detection**, **emotional voice synthesis**, and **final video assembly**, all without manual intervention.

This document describes the system’s architecture, components, data flow, and key implementation details.

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          INPUT LAYER                            │
│  • Manhwa Chapter Images (JPG / PNG)                            │
│  • Configuration Files (YAML)                                   │
│  • API Keys (OpenAI, ElevenLabs, etc.)                          │
└────────────────────┬────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────┐
│                      PROCESSING PIPELINE                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ 1. OCR MODULE                                            │   │
│  │    - Image Preprocessing                                 │   │
│  │    - Text Region Detection                               │   │
│  │    - Character Recognition (Tesseract)                   │   │
│  │    - Structured Text Output                              │   │
│  └────────────────┬─────────────────────────────────────────┘   │
│                   │                                              │
│  ┌────────────────▼─────────────────────────────────────────┐   │
│  │ 2. AI RECAP GENERATION                                  │   │
│  │    - Text Aggregation                                   │   │
│  │    - GPT-4 Summarization                                │   │
│  │    - Narrative Structuring & Context Preservation        │   │
│  └────────────────┬─────────────────────────────────────────┘   │
│                   │                                              │
│  ┌────────────────▼─────────────────────────────────────────┐   │
│  │ 3. PANEL DETECTION & CROPPING                            │   │
│  │    - Panel Boundary Detection (YOLO / OpenCV)            │   │
│  │    - Reading Order Calculation                           │   │
│  │    - Image Cropping & Enhancement                        │   │
│  └────────────────┬─────────────────────────────────────────┘   │
│                   │                                              │
│  ┌────────────────▼─────────────────────────────────────────┐   │
│  │ 4. TEXT-TO-SPEECH SYNTHESIS                              │   │
│  │    - Emotion Detection                                   │   │
│  │    - Voice Selection (Male/Female)                       │   │
│  │    - Audio Generation (ElevenLabs / Coqui)               │   │
│  │    - Prosody & Expression Control                        │   │
│  └────────────────┬─────────────────────────────────────────┘   │
│                   │                                              │
│  ┌────────────────▼─────────────────────────────────────────┐   │
│  │ 5. VIDEO ASSEMBLY                                        │   │
│  │    - Panel Sequencing                                   │   │
│  │    - Audio-Video Synchronization                        │   │
│  │    - Music & Effects Integration                        │   │
│  │    - Final Encoding (FFmpeg)                            │   │
│  └────────────────┬─────────────────────────────────────────┘   │
│                   │                                              │
└───────────────────┼──────────────────────────────────────────────┘
                    │
┌───────────────────▼──────────────────────────────────────────────┐
│                         OUTPUT LAYER                             │
│  • Final MP4 (1080p H.264)                                       │
│  • JSON Metadata & Logs                                          │
│  • Optional YouTube Upload                                       │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. Component Architecture

### 3.1 OCR Module

**Goal:** Extract all textual content from manhwa panels.

**Sub-modules:**
- **ImagePreprocessor** – improves OCR accuracy via:
  - Grayscale conversion  
  - Noise removal  
  - Adaptive thresholding  
  - Contrast and sharpening  
- **TextRegionDetector** – detects dialogue bubbles and text boxes  
- **TextExtractor** – performs OCR using Tesseract (multi-language) and outputs structured JSON with confidence scores  

**Data Flow**
```
Raw Image → Preprocessing → Region Detection → OCR → Structured Text JSON
```

**Output Example**
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
... (truncated for brevity)
