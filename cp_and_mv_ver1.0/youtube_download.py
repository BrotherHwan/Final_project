from pytube import YouTube

def download_video(url, output_path):
    try:
        yt = YouTube(url)
        # 가장 높은 화질의 스트림 가져오기
        stream = yt.streams.filter(progressive=True, file_extension='mp4').first()
        if stream:
            print(f'Downloading {yt.title}...')
            stream.download(output_path)
            print('Download completed!')
        else:
            print('No progressive stream available.')
    except Exception as e:
        print(f'Error: {str(e)}')

# 유튜브 영상 URL
video_url = 'https://www.youtube.com/shorts/jQ26AlHhPTw'

# 다운로드할 경로 지정
output_directory = './videos/'

download_video(video_url, output_directory)
