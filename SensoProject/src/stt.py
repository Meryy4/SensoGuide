import numpy as np
import torch
import whisper
import warnings
from scipy.io import wavfile
from scipy.io.wavfile import WavFileWarning


def get_torch_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = get_torch_device()
model = whisper.load_model("base", device=DEVICE)


def _load_wav_audio(audio_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", WavFileWarning)
        sample_rate, audio = wavfile.read(audio_path)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if np.issubdtype(audio.dtype, np.integer):
        max_val = np.iinfo(audio.dtype).max
        audio = audio.astype(np.float32) / max_val
    else:
        audio = audio.astype(np.float32)

    if sample_rate != 16000:
        duration = audio.shape[0] / sample_rate
        target_len = int(duration * 16000)
        src_times = np.linspace(0, duration, num=audio.shape[0], endpoint=False)
        dst_times = np.linspace(0, duration, num=target_len, endpoint=False)
        audio = np.interp(dst_times, src_times, audio).astype(np.float32)

    return audio


def transcribe_audio(audio_path):
    audio = _load_wav_audio(audio_path)
    result = model.transcribe(audio, fp16=(DEVICE == "cuda"))
    return result["text"]
