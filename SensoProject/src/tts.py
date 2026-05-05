import pyttsx3

def text_to_speech(text, output_path="data/output.wav", speaker="p226"):
    engine = pyttsx3.init()
    engine.setProperty('rate', 160)

    # Force an English voice so it never reads English text with a French accent
    voices = engine.getProperty('voices')
    english_voice = None
    for v in voices:
        lang_str = (v.id + " " + v.name).lower()
        if 'en_us' in lang_str or 'en_gb' in lang_str or 'en-us' in lang_str or 'en-gb' in lang_str or 'english' in lang_str:
            english_voice = v.id
            break
    if english_voice:
        engine.setProperty('voice', english_voice)

    engine.say(text)
    engine.runAndWait()
    engine.stop()
