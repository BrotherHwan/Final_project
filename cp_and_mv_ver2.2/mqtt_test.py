mport paho.mqtt.client as mqtt
import time

# HiveMQ 브로커 정보
broker_address = "broker.hivemq.com"
port = 1883

# 클라이언트 생성
client = mqtt.Client("my_client")  # 클라이언트 ID 지정, 중복되지 않도록 고유한 값을 사용

# 연결 이벤트 처리 함수
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT Broker")
        # 구독 (원하는 토픽을 지정)
        client.subscribe("test/topic")  # 구독할 토픽 설정
    else:
        print(f"Failed to connect, return code: {rc}")

# 메시지 수신 이벤트 처리 함수
def on_message(client, userdata, message):
    print(f"Received message: {message.payload.decode()} on topic {message.topic}")

# 연결 이벤트와 메시지 수신 이벤트 핸들러 등록
client.on_connect = on_connect
client.on_message = on_message

# MQTT 브로커에 연결
client.connect(broker_address, port=port)

# 발행
while True:
    message = input("Enter message to publish (Q to quit): ")
    if message.lower() == 'q':
        break
    client.publish("test/topic", message)  # 발행할 토픽과 메시지 설정
    print(f"Published message: {message}")
    time.sleep(1)  # 1초 딜레이

# 연결 종료
client.disconnect()