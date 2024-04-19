import paho.mqtt.client as mqtt
#client = mqtt_client.Client(mqtt_client.CallbackAPIVersion.VERSION1, client_id)
mqttc = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,"wooseok") # puclisher 이름
mqttc.connect("10.10.52.144", 1883)
mqttc.publish("projectCPMV", "SCORE : 100") # topic, message
