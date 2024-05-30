import whisper
import speech_recognition as sr
from time import time

# Initialize whisper
model = whisper.load_model("small") # tiny/base/small/medium/large

# Initialize recognizer class (for recognizing the speech)
recognizer = sr.Recognizer()

# Set up the microphone
with sr.Microphone() as source:
    print("Please wait. Calibrating microphone for 1 second!")
    # listen for 1 second and create the ambient noise energy level
    recognizer.adjust_for_ambient_noise(source, duration=1)
    print("Microphone is now listening for input up to 5 seconds")
    audio = recognizer.listen(source, phrase_time_limit=5)

    # Write audio to a WAV file
    with open("rec_input.wav", "wb") as f:
        f.write(audio.get_wav_data())

start = time()
# load audio and pad/trim it to fit 30 seconds
audio = whisper.load_audio("rec_input.wav")
audio = whisper.pad_or_trim(audio)

# make log-Mel spectrogram and move to the same device as the model
mel = whisper.log_mel_spectrogram(audio).to(model.device)

# detect the spoken language
_, probs = model.detect_language(mel)
print(f"Detected language: {max(probs, key=probs.get)}")

# decode the audio
options = whisper.DecodingOptions()
result = whisper.decode(model, mel, options)
print(f"time = {time() - start}")

# print the recognized text
print(result.text)