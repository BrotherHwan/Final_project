// #include <UnoWiFiDevEd.h>

/*
 Basic ESP8266 MQTT example
 This sketch demonstrates the capabilities of the pubsub library in combination
 with the ESP8266 board/library.
 It connects to an MQTT server then:
  - publishes "hello world" to the topic "outTopic" every two seconds
  - subscribes to the topic "inTopic", printing out any messages
    it receives. NB - it assumes the received payloads are strings not binary
  - If the first character of the topic "inTopic" is an 1, switch ON the ESP Led,
    else switch it off
 It will reconnect to the server if the connection is lost using a blocking
 reconnect function. See the 'mqtt_reconnect_nonblocking' example for how to
 achieve the same result without blocking the main loop.
 To install the ESP8266 board, (using Arduino 1.6.4+):
  - Add the following 3rd party board manager under "File -> Preferences -> Additional Boards Manager URLs":
       http://arduino.esp8266.com/stable/package_esp8266com_index.json
  - Open the "Tools -> Board -> Board Manager" and click install for the ESP8266"
  - Select your ESP8266 in "Tools -> Board"
*/

#include <ESP8266WiFi.h>
// #include <finclude/math-vector-fortran.h>

#include <PubSubClient.h>

// Update these with values suitable for your network.

const char* ssid = "SEMICON_2.4G";
const char* password = "a1234567890";
const char* mqtt_server = "broker.hivemq.com";

int score = 0;

WiFiClient espClient;
PubSubClient client(espClient);
unsigned long lastMsg = 0;
#define MSG_BUFFER_SIZE	(50)
char msg[MSG_BUFFER_SIZE];
int value = 0;

void setup_wifi() {

  delay(10);
  // We start by connecting to a WiFi network
  Serial.println();
  Serial.print("Connecting to ");
  Serial.println(ssid);

  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  randomSeed(micros());

  Serial.println("");
  Serial.println("WiFi connected");
  Serial.println("IP address: ");
  Serial.println(WiFi.localIP());
}

// void callback(char* topic, byte* payload, unsigned int length) {
//   Serial.print("Message arrived [");
//   Serial.print(topic);
//   Serial.print("] ");
//   for (int i = 0; i < length; i++) {
//     Serial.print((char)payload[i]);
//   }
//   Serial.println();

//   // Switch on the LED if an 1 was received as first character
//   if ((char)payload[0] == '1') {
//     digitalWrite(BUILTIN_LED, LOW);   // Turn the LED on (Note that LOW is the voltage level
//     // but actually the LED is on; this is because
//     // it is active low on the ESP-01)
//   } else {
//     digitalWrite(BUILTIN_LED, HIGH);  // Turn the LED off by making the voltage HIGH
//   }

// }

void reconnect() {
  // Loop until we're reconnected
  while (!client.connected()) {
    
    Serial.print("Attempting MQTT connection...");
    // Create a random client ID
    String clientId = "ESP8266Client-";
    clientId += String(random(0xffff), HEX);
    // Attempt to connect
    if (client.connect(clientId.c_str())) {
      Serial.println("connected");
      // Once connected, publish an announcement...
      // client.publish("CPMV", "hello world");
      // ... and resubscribe
      client.subscribe("CPMV");
    } else {
      Serial.print("failed, rc=");
      Serial.print(client.state());
      Serial.println(" try again in 5 seconds");
      // Wait 5 seconds before retrying
      delay(5000);
    }
  }
}

void setup() {
  pinMode(BUILTIN_LED, OUTPUT);     // Initialize the BUILTIN_LED pin as an output
  Serial.begin(115200);
  setup_wifi();
  // client.setServer(mqtt_server, 1883);
  client.setServer(mqtt_server, 1883);

  client.setCallback(callback);

  pinMode(4, OUTPUT);
  pinMode(0, OUTPUT);
  pinMode(2, OUTPUT);

  digitalWrite(4, 0); // R
  digitalWrite(0, 0); // G
  digitalWrite(2, 0); // B
}

char* substr(const char *src, int m, int n);

void callback(char* topic, byte* payload, unsigned int length) {

  Serial.print("Message arrived [");
  Serial.print(topic);
  Serial.print("] ");

  char* temp_int[3];

  // if ((length > 5) && (((char)payload[0]=='S') ||((char)payload[1]=='c')||((char)payload[2]=='o')||((char)payload[3]=='r')||((char)payload[4]=='e')))
  if (!(strncmp((char*)payload, "wake", 4)))
  {
    Serial.print("wake signal");  
  }
  else if (!(strncmp((char*)payload, "Score", 5)))
  {
   
    score = atoi(substr((char*)payload, 8, 11));// atoi((char*)(payload[8]))*100 + atoi((char*)(payload[9]))*10 + atoi((char*)(payload[10]));
    Serial.print("점수 입력됨 : ");
    Serial.print(score);
    
    analogWrite(4, 255-int(score*2.55));  // R
    analogWrite(0, int(score*2.55));      // G
    analogWrite(2, 0);          // B

    // if (score < 33)       // HP : Red Light
    // {
    //   digitalWrite(4, 1); // R
    //   digitalWrite(0, 0); // G
    //   digitalWrite(2, 0); // B
    // }
    // else if (score < 66)  // HP : Yellow Light 
    // {
    //   digitalWrite(4, 1); // R
    //   digitalWrite(0, 1); // G
    //   digitalWrite(2, 0); // B
    // }
    // else if (score <= 100) // HP : Green Light
    // {
    //   digitalWrite(4, 0); // R
    //   digitalWrite(0, 1); // G
    //   digitalWrite(2, 0); // B
    // }

  } 


  

  // for (int i = 0; i < length; i++) {
  //   Serial.print((char)payload[i]);
  // }
  Serial.println();
// Switch on the LED if an 1 was received as first character
  if ((char)payload[0] == '1') {
    // digitalWrite(14, HIGH);   // Turn the LED on (Note that LOW is the voltage level
    // but actually the LED is on; this is because
    // it is active low on the ESP-01)
  } else {
    // digitalWrite(14, LOW);  // Turn the LED off by making the voltage HIGH
  }
}

char* substr(const char *src, int m, int n)
{
    // 목적지 문자열의 길이를 얻는다.
    int len = n - m;
 
    // 대상에 (len + 1) 문자를 할당합니다(추가 null 문자의 경우 +1).
    char *dest = (char*)malloc(sizeof(char) * (len + 1));
 
    // 소스 문자열에서 m번째와 n번째 인덱스 사이의 문자를 추출합니다.
    // 대상 문자열에 복사
    for (int i = m; i < n && (*(src + i) != '\0'); i++)
    {
        *dest = *(src + i);
        dest++;
    }
 
    // 대상 문자열을 null 종료
    *dest = '\0';
 
    // 목적지 문자열을 반환
    return dest - len;
}


void loop() {

  if (!client.connected()) {
    reconnect();
    
  }

  client.loop();
  // client.subscribe("CPMV");
  
  // unsigned long now = millis();
  // if (now - lastMsg > 2000) {
  //   lastMsg = now;
  //   // ++value;
  //   // snprintf (msg, MSG_BUFFER_SIZE, "hello world #%ld", value);
  //   // Serial.print("Publish message: ");
  //   // Serial.println(msg);
  //   // client.publish("CPMV", msg);
    
  // }
}
