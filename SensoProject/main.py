from src.ocr import extract_text_from_image
from src.stt import transcribe_audio
from src.tts import text_to_speech

image_path = 'data/text.jpg'
audio_path = 'data/output.wav'

def main():
      #text = extract_text_from_image(image_path)
      #print(text)

      #text = "Hello! How are you? How can I help you today?"
      #output_path="data/output2.wav"
      #text_to_speech(text, output_path, speaker="p226")

      audio_path = 'data/output2.wav'
      audio_text = transcribe_audio(audio_path)
      print(audio_text)


if __name__ == "__main__":
    main()
