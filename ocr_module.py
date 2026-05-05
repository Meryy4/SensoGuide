"""
SensoGuide - OCR Module
Real-time text detection + Text-to-Speech output
Integrates with existing YOLOv8 pipeline
"""

import cv2
import numpy as np
import pyttsx3
import threading
import time
import easyocr

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
OCR_LANGUAGES      = ['fr', 'en']   # Languages to detect
OCR_CONFIDENCE     = 0.4            # Minimum confidence threshold (0-1)
TTS_RATE           = 160            # Speech rate (words per minute)
TTS_VOLUME         = 1.0            # Volume (0.0 to 1.0)
OCR_COOLDOWN       = 3.0            # Seconds between auto-OCR triggers
MIN_TEXT_LENGTH    = 3              # Ignore text shorter than N characters
PREPROCESS_RESIZE  = 1.5            # Upscale factor for better OCR accuracy


# ──────────────────────────────────────────────
# OCR Engine
# ──────────────────────────────────────────────
class SensoOCR:
    def __init__(self):
        print("[SensoOCR] Loading EasyOCR model...")
        self.reader = easyocr.Reader(OCR_LANGUAGES, gpu=False)
        print("[SensoOCR] Model loaded.")

        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', TTS_RATE)
        self.tts_engine.setProperty('volume', TTS_VOLUME)

        self._tts_lock = threading.Lock()
        self._last_ocr_time = 0.0
        self._last_spoken_text = ""
        self._speaking = False

    # ── Preprocessing ──────────────────────────
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Enhance frame for better OCR accuracy:
          1. Upscale
          2. Grayscale
          3. CLAHE contrast enhancement
          4. Adaptive threshold
        """
        h, w = frame.shape[:2]
        upscaled = cv2.resize(
            frame,
            (int(w * PREPROCESS_RESIZE), int(h * PREPROCESS_RESIZE)),
            interpolation=cv2.INTER_CUBIC
        )
        gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        # Adaptive threshold for varying lighting conditions
        binary = cv2.adaptiveThreshold(
            enhanced, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        return binary

    # ── OCR ────────────────────────────────────
    def read_text(self, frame: np.ndarray) -> list[dict]:
        """
        Run OCR on a frame.
        Returns a list of dicts: {text, confidence, bbox}
        """
        processed = self.preprocess(frame)
        results = self.reader.readtext(processed)

        detections = []
        for (bbox, text, confidence) in results:
            text = text.strip()
            if confidence >= OCR_CONFIDENCE and len(text) >= MIN_TEXT_LENGTH:
                # Scale bbox back to original frame coordinates
                scale = 1.0 / PREPROCESS_RESIZE
                scaled_bbox = [[int(pt[0] * scale), int(pt[1] * scale)] for pt in bbox]
                detections.append({
                    "text": text,
                    "confidence": round(confidence, 2),
                    "bbox": scaled_bbox
                })

        return detections

    # ── TTS ────────────────────────────────────
    def speak(self, text: str, force: bool = False):
        """
        Speak text in a non-blocking thread.
        Skips if same text was spoken recently (unless force=True).
        """
        if not force and text == self._last_spoken_text:
            return
        if self._speaking:
            return

        self._last_spoken_text = text

        def _run():
            self._speaking = True
            with self._tts_lock:
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            self._speaking = False

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

    # ── Draw Results ───────────────────────────
    def draw_results(self, frame: np.ndarray, detections: list[dict]) -> np.ndarray:
        """
        Draw bounding boxes and text on the frame.
        """
        overlay = frame.copy()
        for det in detections:
            bbox = det["bbox"]
            text = det["text"]
            conf = det["confidence"]

            pts = np.array(bbox, dtype=np.int32)

            # Semi-transparent background box
            cv2.fillPoly(overlay, [pts], (0, 180, 255))
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

            # Bounding box border
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 180, 255), thickness=2)

            # Text label
            x, y = bbox[0]
            label = f"{text} ({int(conf * 100)}%)"
            cv2.putText(
                frame, label,
                (x, max(y - 8, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2, cv2.LINE_AA
            )
        return frame

    # ── Main integration method ─────────────────
    def process_frame(self, frame: np.ndarray, trigger: bool = False) -> np.ndarray:
        """
        Call this from your YOLOv8 loop.

        Args:
            frame:   Current camera frame (BGR)
            trigger: True when user presses OCR key (e.g. 't')

        Returns:
            Annotated frame with OCR overlays
        """
        now = time.time()
        cooldown_ok = (now - self._last_ocr_time) >= OCR_COOLDOWN

        if not (trigger or cooldown_ok):
            return frame

        self._last_ocr_time = now
        detections = self.read_text(frame)

        if detections:
            # Build spoken sentence
            all_text = " — ".join([d["text"] for d in detections])
            self.speak(all_text, force=trigger)
            frame = self.draw_results(frame, detections)
            print(f"[SensoOCR] Detected: {all_text}")
        elif trigger:
            self.speak("Aucun texte détecté.", force=True)

        return frame
