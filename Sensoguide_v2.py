import os
import subprocess
import tempfile
import threading
import time
from collections import Counter, deque
from datetime import datetime
from itertools import count

import AVFoundation
import cv2
import numpy as np
import Quartz
import torch
import Vision
from Foundation import NSURL
from groq import Groq
from ultralytics import YOLO

from SensoProject.src.stt import transcribe_audio


# ═══════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════
YOLO_MODEL_PATH      = "best.pt"
GROQ_API_KEY         = os.environ.get('GROQ_API_KEY', '')
OCR_CONFIDENCE       = 0.5
OCR_LANGUAGES_LATIN  = ["fr-FR", "en-US"]
OCR_LANGUAGES_ARABIC = ["ar-SA"]
ASK_RECORD_SECONDS   = 5.0
MEMORY_LIMIT         = 20
WINDOW_NAME          = "SensoGuide"
YOLO_DEVICE_OVERRIDE = os.getenv("SENSO_YOLO_DEVICE", "").strip().lower()
TTS_VOICE            = "Samantha"   # macOS English voice

PLURALS = {
    "person": "persons", "chair": "chairs", "toothbrush": "toothbrushes",
    "knife": "knives", "bottle": "bottles", "cup": "cups", "spoon": "spoons",
    "bench": "benches", "fork": "forks", "bus": "buses", "bicycle": "bicycles",
    "truck": "trucks", "motorcycle": "motorcycles", "oven": "ovens", "bed": "beds",
    "cat": "cats", "traffic light": "traffic lights", "currency": "currencies",
    "stop sign": "stop signs", "car": "cars", "barriers": "barriers",
    "path holes": "path holes", "stairs": "stairs", "train": "trains",
    "bin": "bins", "blind stick": "blind sticks", "men sign": "men signs",
    "cell phone": "cell phones", "women sign": "women signs", "tap": "taps",
}


# ═══════════════════════════════════════════════════════════
#  SPEAKER — uses macOS `say` command, zero deadlock risk
# ═══════════════════════════════════════════════════════════
class Speaker:
    def __init__(self):
        self._proc  = None
        self._lock  = threading.Lock()

    def speak(self, text, block=False):
        """Fire macOS say. block=True waits for completion."""
        if not text or not text.strip():
            return
        with self._lock:
            # Kill any running speech first
            if self._proc and self._proc.poll() is None:
                self._proc.terminate()
                self._proc.wait()
            self._proc = subprocess.Popen(
                ["say", "-v", TTS_VOICE, "-r", "180", text],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        if block:
            self._proc.wait()

    def wait_done(self, timeout=15):
        with self._lock:
            p = self._proc
        if p:
            try:
                p.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                p.terminate()

    @property
    def is_busy(self):
        with self._lock:
            return self._proc is not None and self._proc.poll() is None


# ═══════════════════════════════════════════════════════════
#  SYSTEM STATE
# ═══════════════════════════════════════════════════════════
class SystemState:
    def __init__(self):
        self.mode             = "idle"
        self.is_listening     = False
        self.ask_status       = ""
        self.latest_objects   = ""
        self.latest_text      = ""
        self.latest_sentence  = ""
        self.latest_summary   = ""
        self.latest_frame     = None
        self.frame_lock       = threading.Lock()
        self.memory           = deque(maxlen=MEMORY_LIMIT)
        self.memory_lock      = threading.Lock()
        self.window_clicked   = False
        self.last_text_raw    = ""
        # ONE action at a time — set with a lock to avoid race on key hold
        self._action_lock     = threading.Lock()
        self._action_running  = False

    def try_start_action(self, mode_name):
        """Atomically check-and-set. Returns True only once per action."""
        with self._action_lock:
            if self._action_running:
                return False
            self._action_running = True
            self.mode = mode_name
            return True

    def end_action(self):
        with self._action_lock:
            self._action_running = False
            self.mode = "idle"

    @property
    def action_running(self):
        with self._action_lock:
            return self._action_running


speaker = Speaker()
client  = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
state   = SystemState()
detector_model           = None
runtime_detection_device = None
ask_counter              = count(1)


# ═══════════════════════════════════════════════════════════
#  DEVICE
# ═══════════════════════════════════════════════════════════
def get_compute_device():
    if torch.cuda.is_available():         return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"

def get_detection_device():
    if YOLO_DEVICE_OVERRIDE in {"cpu","cuda","mps"}: return YOLO_DEVICE_OVERRIDE
    d = get_compute_device()
    return "cpu" if d == "mps" else d


# ═══════════════════════════════════════════════════════════
#  MEMORY
# ═══════════════════════════════════════════════════════════
def ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def add_memory(kind, content, timestamp=None):
    if not content or not content.strip(): return
    with state.memory_lock:
        state.memory.append({
            "timestamp": timestamp or ts(),
            "kind": kind,
            "content": content.strip()
        })

def format_memory():
    with state.memory_lock:
        entries = list(state.memory)
    if not entries: return "No memory yet."
    return "\n".join(
        f"- [{e['timestamp']}] {e['kind']}: {e['content']}" for e in entries)


# ═══════════════════════════════════════════════════════════
#  SENTENCE BUILDER
# ═══════════════════════════════════════════════════════════
def build_detection_sentence(label_list):
    if not label_list: return "Clear path."
    counts = Counter(label_list)
    parts = []
    for label, n in counts.items():
        ll = label.lower()
        parts.append(f"1 {ll}" if n == 1 else f"{n} {PLURALS.get(ll, ll+'s')}")
    if len(parts) == 1:
        return f"I see {parts[0]} in front of you."
    return f"I see {', '.join(parts[:-1])} and {parts[-1]} in front of you."


# ═══════════════════════════════════════════════════════════
#  UI
# ═══════════════════════════════════════════════════════════
def handle_window_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        state.window_clicked = True

def draw_overlay(frame):
    overlay = frame.copy()
    if state.is_listening:
        status = f"ASK — {state.ask_status}"
    elif state.action_running:
        status = f"{state.mode.upper()} — processing…"
    else:
        status = "IDLE — press D / T / A"
    lines = [
        f"Status: {status}",
        "D = detect   T = read text   A = ask   Q = quit",
    ]
    if not state.window_clicked:
        lines.append("Click window to enable keyboard shortcuts")
    h = min(36 + 28 * len(lines), frame.shape[0] - 10)
    cv2.rectangle(overlay, (10,10), (frame.shape[1]-10, h), (20,20,20), -1)
    frame[:] = cv2.addWeighted(overlay, 0.72, frame, 0.28, 0)
    y = 38
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (20,y), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    (255,255,255) if i==0 else (0,255,255), 2)
        y += 28
    return frame

def show_startup_error_screen(lines):
    frame = np.zeros((480,900,3), dtype=np.uint8)
    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, handle_window_mouse)
    while True:
        screen = frame.copy()
        cv2.putText(screen,"SensoGuide Startup Error",(20,50),
                    cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255),2)
        y = 100
        for line in lines:
            cv2.putText(screen,line,(20,y),cv2.FONT_HERSHEY_SIMPLEX,0.65,(255,255,255),2)
            y += 35
        cv2.imshow(WINDOW_NAME, screen)
        if cv2.waitKey(50) & 0xFF == ord("q"): break
    cv2.destroyAllWindows()


# ═══════════════════════════════════════════════════════════
#  CAMERA
# ═══════════════════════════════════════════════════════════
def open_camera():
    backends = []
    if hasattr(cv2,"CAP_AVFOUNDATION"):
        backends.append(("avfoundation", cv2.CAP_AVFOUNDATION))
    backends.append(("default", None))
    for name, backend in backends:
        for _ in range(3):
            cap = cv2.VideoCapture(0) if backend is None else cv2.VideoCapture(0, backend)
            if not cap.isOpened():
                cap.release(); time.sleep(0.4); continue
            for _ in range(12):
                ret, f = cap.read()
                if ret and f is not None and f.size > 0:
                    print(f"[Camera] Opened with backend: {name}")
                    return cap
                time.sleep(0.1)
            cap.release(); time.sleep(0.4)
    return None


# ═══════════════════════════════════════════════════════════
#  OCR
# ═══════════════════════════════════════════════════════════
def _frame_to_cg(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, c = rgb.shape
    p = Quartz.CGDataProviderCreateWithData(None, rgb.tobytes(), h*w*c, None)
    return Quartz.CGImageCreate(w,h,8,24,w*c,
        Quartz.CGColorSpaceCreateDeviceRGB(),
        Quartz.kCGBitmapByteOrderDefault,p,None,False,
        Quartz.kCGRenderingIntentDefault)

def _vision_ocr(cgimage, languages):
    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cgimage,{})
    req = Vision.VNRecognizeTextRequest.alloc().init()
    req.setRecognitionLanguages_(languages)
    ok, err = handler.performRequests_error_([req], None)
    if not ok: raise RuntimeError(f"Vision OCR failed: {err}")
    return [
        obs.topCandidates_(1)[0].string()
        for obs in (req.results() or [])
        if obs.confidence() > OCR_CONFIDENCE and obs.topCandidates_(1)
    ]

def summarize_text(raw):
    if not raw.strip(): return ""
    if client is None: return "Groq is not configured."
    try:
        r = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role":"user","content":
                "The following text was found in front of the user. "
                "Describe in one short English sentence what the user is looking at. "
                "Ignore gibberish.\n\nTEXT: " + raw}])
        return r.choices[0].message.content.strip()
    except Exception as e:
        print(f"[Groq OCR] {e}")
        return "I see some text but cannot identify the object."


# ═══════════════════════════════════════════════════════════
#  AUDIO RECORDING
# ═══════════════════════════════════════════════════════════
def record_audio(duration=ASK_RECORD_SECONDS):
    fd, path = tempfile.mkstemp(suffix=".wav", prefix="sensoguide_")
    os.close(fd)
    settings = {
        AVFoundation.AVFormatIDKey:               AVFoundation.kAudioFormatLinearPCM,
        AVFoundation.AVSampleRateKey:             16000.0,
        AVFoundation.AVNumberOfChannelsKey:       1,
        AVFoundation.AVLinearPCMBitDepthKey:      16,
        AVFoundation.AVLinearPCMIsFloatKey:       False,
        AVFoundation.AVLinearPCMIsBigEndianKey:   False,
        AVFoundation.AVLinearPCMIsNonInterleaved: False,
    }
    url = NSURL.fileURLWithPath_(path)
    rec, err = AVFoundation.AVAudioRecorder.alloc().initWithURL_settings_error_(
        url, settings, None)
    if rec is None: raise RuntimeError(f"AVAudioRecorder failed: {err}")
    if not rec.prepareToRecord(): raise RuntimeError("Recorder could not prepare.")
    if not rec.record(): raise RuntimeError("Recording did not start.")
    time.sleep(duration)
    rec.stop()
    return path


# ═══════════════════════════════════════════════════════════
#  ACTIONS  (each runs in its own daemon thread)
# ═══════════════════════════════════════════════════════════
def action_detect():
    if not state.try_start_action("detection"): return
    try:
        with state.frame_lock:
            frame = state.latest_frame.copy() if state.latest_frame is not None else None
        if frame is None:
            speaker.speak("No camera frame available.", block=True); return

        results  = detector_model(frame, verbose=False, device=runtime_detection_device)
        names    = results[0].names
        labels   = [names[int(c)] for c in results[0].boxes.cls.cpu().numpy()]
        sentence = build_detection_sentence(labels)
        state.latest_objects  = ", ".join(labels)
        state.latest_sentence = sentence
        add_memory("detection", sentence)

        annotated = results[0].plot()
        with state.frame_lock:
            state.latest_frame = annotated

        print(f"[Detect] {sentence}")
        speaker.speak(sentence, block=True)

    except Exception as e:
        print(f"[Detect Error] {e}")
        if runtime_detection_device != "cpu":
            globals()["runtime_detection_device"] = "cpu"
        speaker.speak("Detection failed.", block=True)
    finally:
        state.end_action()


def action_text():
    if not state.try_start_action("text"): return
    try:
        with state.frame_lock:
            frame = state.latest_frame.copy() if state.latest_frame is not None else None
        if frame is None:
            speaker.speak("No camera frame available.", block=True); return

        cg  = _frame_to_cg(frame)
        raw = " ".join(_vision_ocr(cg, OCR_LANGUAGES_LATIN + OCR_LANGUAGES_ARABIC))
        state.latest_text = raw
        print(f"[Text] OCR: {raw!r}")

        if not raw:
            speaker.speak("I cannot see any text here.", block=True); return

        if raw == state.last_text_raw and state.latest_summary:
            speaker.speak(state.latest_summary, block=True); return

        state.last_text_raw = raw
        add_memory("ocr_text", raw)
        desc = summarize_text(raw)
        state.latest_summary = desc
        if desc: add_memory("scene_summary", desc)

        print(f"[Text] {desc}")
        speaker.speak(desc or raw, block=True)

    except Exception as e:
        print(f"[Text Error] {e}")
        speaker.speak("Text reading failed.", block=True)
    finally:
        state.end_action()


def action_ask():
    if not state.try_start_action("ask"): return
    audio_path = None
    try:
        state.is_listening = True
        state.ask_status   = "listening…"

        # ── Speak FIRST, then record ──────────────────────
        speaker.speak("I am listening.", block=True)
        time.sleep(0.3)   # tiny gap so mic doesn't catch TTS tail

        state.ask_status = "recording…"
        print("[Ask] Recording…")
        audio_path = record_audio()

        state.ask_status = "transcribing…"
        question = transcribe_audio(audio_path).strip()
        print(f"[Ask] Question: {question!r}")

        if not question:
            speaker.speak("I did not hear anything.", block=True); return

        # Quick YOLO snapshot for context
        state.ask_status = "thinking…"
        with state.frame_lock:
            snap = state.latest_frame.copy() if state.latest_frame is not None else None
        if snap is not None:
            try:
                r = detector_model(snap, verbose=False, device=runtime_detection_device)
                names  = r[0].names
                labels = [names[int(c)] for c in r[0].boxes.cls.cpu().numpy()]
                state.latest_sentence = build_detection_sentence(labels)
                state.latest_objects  = ", ".join(labels)
                add_memory("detection", state.latest_sentence)
            except Exception as e:
                print(f"[Ask scene] {e}")

        if client is None:
            speaker.speak("Groq is not configured.", block=True); return

        prompt = f"""Current observation:
- Detection: {state.latest_sentence or 'Unknown'}
- OCR text:  {state.latest_text or 'None'}
- Summary:   {state.latest_summary or 'None'}

Timestamped memory:
{format_memory()}

User question (may be French, Arabic, Spanish, or English): "{question}"

Instructions:
- ALWAYS answer in English only.
- Be concise: 1-3 sentences maximum.
- Do not repeat the question.
- If context is insufficient, say you are not sure.
"""
        res = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role":"user","content":prompt}],
        )
        answer = res.choices[0].message.content.strip()
        print(f"[Ask] Answer: {answer}")
        state.ask_status = "speaking…"
        speaker.speak(answer, block=True)

    except Exception as e:
        print(f"[Ask Error] {e}")
        speaker.speak("Ask mode failed. Check microphone permissions.", block=True)
    finally:
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)
        state.is_listening = False
        state.ask_status   = ""
        state.end_action()
        print("[Ask] Done.")


# ═══════════════════════════════════════════════════════════
#  MAIN LOOP
# ═══════════════════════════════════════════════════════════
def main():
    global detector_model, runtime_detection_device

    runtime_detection_device = get_detection_device()
    detector_model = YOLO(YOLO_MODEL_PATH)
    cap = open_camera()

    if cap is None:
        speaker.speak("I could not open the camera.", block=True)
        show_startup_error_screen([
            "Camera access is blocked or unavailable.",
            "Allow camera access in macOS Privacy and Security settings.",
        ])
        return

    print(f"[Compute] torch={get_compute_device()}  yolo={runtime_detection_device}")
    print("[SensoGuide] D=Detect | T=Read text | A=Ask | Q=Quit")

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, handle_window_mouse)
    speaker.speak("SensoGuide ready. Press D to detect, T to read text, or A to ask.", block=True)

    last_key_time = 0   # debounce: ignore key repeats within 0.5s

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05); continue

        # Only refresh latest_frame when no action is writing an annotated result
        if not state.action_running:
            with state.frame_lock:
                state.latest_frame = frame.copy()

        display = state.latest_frame.copy() if state.latest_frame is not None else frame
        cv2.imshow(WINDOW_NAME, draw_overlay(display))

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        now = time.time()
        # Accept key only when idle AND debounced (prevents key-hold repeat)
        if key != 0xFF and not state.action_running and not state.is_listening:
            if now - last_key_time > 0.5:
                last_key_time = now
                if key == ord("d"):
                    threading.Thread(target=action_detect, daemon=True).start()
                elif key == ord("t"):
                    threading.Thread(target=action_text, daemon=True).start()
                elif key == ord("a"):
                    threading.Thread(target=action_ask, daemon=True).start()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()