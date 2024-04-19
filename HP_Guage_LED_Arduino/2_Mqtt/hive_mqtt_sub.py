import paho.mqtt.client as mqtt


def on_connect(self,client, userdata, flags, rc):
    if rc == 0:
        print("connected OK")
    else:
        print("Bad connection Returned code=", rc)


def on_disconnect(self,client, userdata, flags, rc=0):
    print(str(rc))


def on_subscribe(self,client, userdata, mid, granted_qos):
    print("subscribed: " + str(mid) + " " + str(granted_qos))


def on_message(client, userdata, msg):
    print(str(msg.payload.decode("utf-8")))


# 새로운 클라이언트 생성
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
# 콜백 함수 설정 on_connect(브로커에 접속), on_disconnect(브로커에 접속중료), on_subscribe(topic 구독),
# on_message(발행된 메세지가 들어왔을 때)
client.on_connect = on_connect
client.on_disconnect = on_disconnect
client.on_subscribe = on_subscribe
client.on_message = on_message
# address : localhost, port: 1883 에 연결
client.connect('broker.hivemq.com', 1883)
# common topic 으로 메세지 발행
client.subscribe('CPMV', 1)
client.loop_forever()
