import paho.mqtt.client as mqtt
import time
# MQTT 브로커의 주소와 포트
broker_address = "broker.hivemq.com"
port = 1883

# 클라이언트 식별자
client_id = "client1"

# MQTT 클라이언트 객체 생성
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,client_id)

# MQTT 브로커에 연결
client.connect(broker_address, port)

while True:
    key = input()
    if key == "q":
        break    
    topic = "CPMV"
    message = key
    client.publish(topic, message)
#topic = "CPMV"
#message = "test : 10000000"
#client.publish(topic, message)
#time.sleep(0.5)
