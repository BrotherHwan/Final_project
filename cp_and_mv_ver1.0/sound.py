import whisper
import speech_recognition as sr
from time import time


class sound_reco():
    def __init__(self, queue):
        # Initialize whisper
        self.model = whisper.load_model("medium") # tiny/base/small/medium/large
        #model = whisper.load_model("base", "cpu") # tiny/base/small/medium/large
        # Initialize recognizer class (for recognizing the speech)
        self.recognizer = sr.Recognizer()
        
        with sr.Microphone() as source:
            print("Please wait. Calibrating microphone for 1 second!")
            # listen for 1 second and create the ambient noise energy level
            self.recognizer.adjust_for_ambient_noise(source, duration=1)

    def tts(self, queue):
        # Set up the microphone
        with sr.Microphone() as source:
            print("Microphone is now listening for input up to 10 seconds")
            audio = self.recognizer.listen(source, phrase_time_limit=10)
            # Write audio to a WAV file
            with open("rec_input.wav", "wb") as f:
                f.write(audio.get_wav_data())

        # load audio and pad/trim it to fit 30 seconds
        audio = whisper.load_audio("rec_input.wav")
        audio = whisper.pad_or_trim(audio)
        # make log-Mel spectrogram and move to the same device as the model
        mel = whisper.log_mel_spectrogram(audio).to(model.device)

        # decode the audio
        options = whisper.DecodingOptions(language='ko')
        result = whisper.decode(model, mel, options)

        # print the recognized text
        print(result.text)
        
        queue.put({"flag" : result.text})
        
        
if __name__ == "__main__":
    sound_reco = sound_reco()
    while True:
        key = input()
        if key == "q":
            break
        
        sound_reco.tts()
        
        