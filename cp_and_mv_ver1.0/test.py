import multiprocessing
from moviepy.editor import VideoFileClip
import pygame


def play_sound_in_subprocess(sound_file):
    # 서브프로세스에서 소리 재생을 위해 초기화
    pygame.mixer.init()
    
    # 소리 파일 로드
    sound = pygame.mixer.Sound(sound_file)
    
    # 소리 재생
    sound.play()
    
    # 재생이 끝날 때까지 대기
    while pygame.mixer.get_busy():
        continue


def process_video(video_file, sound_file):
    # 비디오 파일 로드
    video = VideoFileClip(video_file)
    video.preview()
    # 비디오에 소리 추가
    video.audio.write_audiofile(sound_file)
    
    # 결과 비디오 파일 저장
    

def main():
    video_file = './m.mp4'
    sound_file = 'output_sound.wav'

    # 멀티프로세스로 동작하도록 설정
    sound_process = multiprocessing.Process(target=play_sound_in_subprocess, args=(sound_file,))
    video_process = multiprocessing.Process(target=process_video, args=(video_file, sound_file))

    # 서브프로세스 시작
    sound_process.start()
    video_process.start()

    # 프로세스 종료 대기
    sound_process.join()
    video_process.join()

    print("All processes completed.")


if __name__ == "__main__":
    main()