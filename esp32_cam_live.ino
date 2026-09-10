#include <WiFi.h>
#include <WebServer.h>
#include <ESPmDNS.h>
#include "esp_camera.h"

const char* ssid = "vivo T4";
const char* password = "12345679";
const char* apPassword = "ESP32Cam123";
const char* mdnsName = "esp32cam";

WebServer server(80);
bool accessPointMode = false;
unsigned long lastReconnectAttempt = 0;
unsigned long framesSent = 0;
unsigned long bytesSent = 0;
unsigned long lastStats = 0;

// AI Thinker ESP32-CAM pin configuration
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

void handleRoot() {
  const char html[] = R"rawl(
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ESP32-CAM</title>
  <style>
    body { margin: 0; background: #101827; color: white; font-family: Arial, sans-serif; text-align: center; }
    h2 { margin: 20px; }
    img { width: 95%; max-width: 800px; border: 2px solid #4b5563; border-radius: 10px; }
  </style>
</head>
<body>
  <h2>ESP32-CAM Live Stream</h2>
  <img src="/live">
</body>
</html>
)rawl";

  server.send(200, "text/html", html);
}

void handleLive() {
  server.sendHeader("Cache-Control", "no-cache");
  server.sendHeader("Pragma", "no-cache");
  server.sendHeader("Access-Control-Allow-Origin", "*");
  server.setContentLength(CONTENT_LENGTH_UNKNOWN);
  server.send(200, "multipart/x-mixed-replace; boundary=ESP32CAM", "");

  WiFiClient client = server.client();

  while (client.connected()) {
    camera_fb_t* frame = esp_camera_fb_get();
    if (frame == nullptr) {
      Serial.println("Camera capture failed");
      break;
    }

    client.print("--ESP32CAM\r\n");
    client.print("Content-Type: image/jpeg\r\n");
    client.print("Content-Length: ");
    client.print(frame->len);
    client.print("\r\n\r\n");
    client.write(frame->buf, frame->len);
    client.print("\r\n");

    framesSent++;
    bytesSent += frame->len;

    esp_camera_fb_return(frame);
    delay(50);
  }

  client.stop();
}

bool setupCamera() {
  camera_config_t config = {};
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = FRAMESIZE_VGA;
  config.jpeg_quality = 12;
  config.fb_count = psramFound() ? 2 : 1;
  config.grab_mode = CAMERA_GRAB_LATEST;
  config.fb_location = psramFound() ? CAMERA_FB_IN_PSRAM : CAMERA_FB_IN_DRAM;

  if (!psramFound()) {
    config.frame_size = FRAMESIZE_QVGA;
  }

  esp_err_t result = esp_camera_init(&config);
  if (result != ESP_OK) {
    Serial.print("Camera initialization failed: 0x");
    Serial.println(result, HEX);
    return false;
  }

  return true;
}

void startAccessPoint() {
  accessPointMode = true;
  WiFi.mode(WIFI_AP);
  WiFi.softAP("ESP32-CAM-Safety", apPassword);
  Serial.println("Wi-Fi AP mode enabled");
  Serial.print("AP IP: ");
  Serial.println(WiFi.softAPIP());
}

void setupWiFi() {
  WiFi.mode(WIFI_STA);
  WiFi.setHostname(mdnsName);
  WiFi.begin(ssid, password);
  Serial.print("Connecting to Wi-Fi");

  unsigned long started = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - started < 20000) {
    delay(500);
    Serial.print(".");
  }

  Serial.println();
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("Wi-Fi connected in station mode");
    Serial.print("ESP32-CAM IP: ");
    Serial.println(WiFi.localIP());
    if (MDNS.begin(mdnsName)) {
      MDNS.addService("http", "tcp", 80);
      Serial.println("mDNS: http://esp32cam.local/");
    } else {
      Serial.println("mDNS could not be started");
    }
  } else {
    Serial.println("Wi-Fi connection timed out");
    startAccessPoint();
  }
}

void maintainWiFi() {
  if (accessPointMode || WiFi.status() == WL_CONNECTED) {
    return;
  }

  if (millis() - lastReconnectAttempt < 10000) {
    return;
  }

  lastReconnectAttempt = millis();
  Serial.println("Wi-Fi disconnected; reconnecting");
  WiFi.disconnect();
  WiFi.begin(ssid, password);
}

void printDiagnostics() {
  if (millis() - lastStats < 10000) {
    return;
  }

  lastStats = millis();
  Serial.print("Heap: ");
  Serial.print(ESP.getFreeHeap());
  Serial.print(" bytes, RSSI: ");
  Serial.print(accessPointMode ? 0 : WiFi.RSSI());
  Serial.print(" dBm, frames: ");
  Serial.print(framesSent);
  Serial.print(", bytes: ");
  Serial.println(bytesSent);
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  if (!setupCamera()) {
    while (true) {
      delay(1000);
    }
  }

  setupWiFi();
  server.on("/", HTTP_GET, handleRoot);
  server.on("/live", HTTP_GET, handleLive);
  server.begin();

  Serial.println("ESP32-CAM server started");
  Serial.print("Open http://");
  Serial.print(WiFi.localIP());
  Serial.println("/live");
}

void loop() {
  server.handleClient();
  maintainWiFi();
  printDiagnostics();
}
