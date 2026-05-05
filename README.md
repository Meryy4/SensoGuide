# SensoGuide

Real-time object detection and scene description system designed to assist visually impaired users. Built with YOLOv8 (custom 35-class model), Groq LLM for intelligent scene narration, Apple Vision OCR, Whisper STT, and pyttsx3 TTS.

## Features

- **Real-time object detection** — Custom YOLOv8 model fine-tuned on 35 everyday object classes (mAP50 ≈ 0.86)
- **Scene description** — LLM-powered natural language descriptions of detected objects via Groq API
- **OCR** — Text reading using Apple Vision framework (Latin + Arabic) and EasyOCR fallback
- **Speech-to-Text** — Voice commands via OpenAI Whisper
- **Text-to-Speech** — Audio feedback with pyttsx3
- **Mobile app** — React Native / Expo companion app (`SensoGuideApp/`)
- **API server** — Flask-based API for the mobile app to connect to the detection backend

## Project Structure

```
├── Sensoguide_v2.py        # Main desktop application (webcam detection + TTS)
├── api_server.py           # API server for mobile app
├── ocr_module.py           # Apple Vision OCR module
├── ocr_image.py            # OCR image utility
├── best.pt                 # YOLOv8 model weights (not tracked — see Setup)
├── requirements.txt        # Python dependencies (detection)
├── requirements_ocr.txt    # Python dependencies (OCR)
├── SensoProject/           # Core modules (TTS, STT, OCR)
│   └── src/
│       ├── tts.py
│       ├── stt.py
│       └── ocr.py
├── SensoGuideApp/          # React Native / Expo mobile app
│   ├── app/
│   ├── components/
│   └── package.json
└── data/                   # Sample audio files
```

## Setup

### Prerequisites

- Python 3.10+
- Node.js 18+ (for the mobile app)
- macOS recommended (Apple Vision OCR uses native frameworks)

### Backend

```bash
# Clone the repo
git clone https://github.com/<your-username>/SensoGuide.git
cd SensoGuide

# Install dependencies
pip install -r requirements.txt
pip install -r requirements_ocr.txt

# Set your Groq API key
export GROQ_API_KEY="your_groq_api_key_here"

# Download model weights (best.pt) and place in project root
# (contact the team or check Releases for the weights file)

# Run the desktop app
python Sensoguide_v2.py
```

### Mobile App

```bash
cd SensoGuideApp
npm install
npx expo start
```

## Environment Variables

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | API key for Groq LLM scene description |

See `.env.example` for reference.

## Controls (Desktop)

- Webcam opens automatically on launch
- Press **Q** to quit

## Tech Stack

- **Detection**: YOLOv8 (Ultralytics), OpenCV, PyTorch
- **LLM**: Groq API (scene narration)
- **OCR**: Apple Vision framework, EasyOCR
- **STT**: OpenAI Whisper
- **TTS**: pyttsx3
- **Mobile**: React Native, Expo, TypeScript
- **API**: Flask / FastAPI

## Authors

ENSEEIHT — INP Toulouse

## License

This project was developed for academic purposes.
