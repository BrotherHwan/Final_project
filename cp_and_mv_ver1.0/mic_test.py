import speech_recognition as sr
from time import time

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
    #with open("rec_input.wav", "wb") as f:
    #    f.write(audio.get_wav_data())

start = time()
# load audio and pad/trim it to fit 30 seconds
text = recognizer.recognize_google(audio, language='ko-KR')
print(f"time: {time() - start}")
print( text )