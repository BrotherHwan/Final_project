import whisper
import speech_recognition as sr
from time import time
import sys


class sound_reco():
    def __init__(self):
        self.confirm = ["선택", '실행', '확인', '시작']
        # Initialize whisper
        self.model = whisper.load_model("small")  # tiny/base/small/medium/large
        # model = whisper.load_model("base", "cpu") # tiny/base/small/medium/large
        # Initialize recognizer class (for recognizing the speech)
        self.recognizer = sr.Recognizer()

    def tts(self):
        # Set up the microphone
        with sr.Microphone() as source:
            print("감도 조정중입니다. 잠시 기다려 주세요!")
            # listen for 1 second and create the ambient noise energy level
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            # print("Microphone is now listening for input up to 10 seconds")
            print("음성 명령을 내려주세요(댄스, 야구, 농구, 축구, 골프 중 하나를 골라서 이야기해주세요)")
            audio = self.recognizer.listen(source, phrase_time_limit=3)
            # Write audio to a WAV file
            with open("rec_input.wav", "wb") as f:
                f.write(audio.get_wav_data())

        # load audio and pad/trim it to fit 30 seconds
        audio = whisper.load_audio("rec_input.wav")
        audio = whisper.pad_or_trim(audio)
        # make log-Mel spectrogram and move to the same device as the model
        mel = whisper.log_mel_spectrogram(audio).to(self.model.device)

        # decode the audio
        options = whisper.DecodingOptions(language='ko')
        result = whisper.decode(self.model, mel, options)

        # print the recognized text
        
        if "댄스" in result.text:
            print("댄스을 선택하였습니다..")
            return "댄스"
        elif "농구" in result.text:
            print("농구을 선택하였습니다..")
            return "농구"
        elif "축구" in result.text:
            print("축구을 선택하였습니다..")
            return "축구"
        elif "야구" in result.text:
            print("야구를 선택하였습니다..")
            return "야구"
        elif "골프" in result.text:
            print("골프를 선택하였습니다..")
            return "골프"
        elif "종료" in result.text:
            return "종료"          
        elif any(i in result.text for i in self.confirm):
            print("시작인식했음.")
            return "start"

 
if __name__ == "__main__":
    sound_rec = sound_reco()
    while True:
        result = sound_rec.tts()
        if result == "종료":
            sys.exit()
            break
