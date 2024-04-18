from moviepy.editor import *

# 동영상 파일 로드
video = VideoFileClip('m.mp4')

# 동영상 재생 (영상과 소리 포함)
# video.preview()
video.audio()
