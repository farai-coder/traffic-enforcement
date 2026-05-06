/*
 * Simple Single Traffic Light Test - ESP32
 * =========================================
 * Tests one traffic light by cycling: RED -> YELLOW -> GREEN
 * Use this to verify your wiring before running the full controller.
 *
 * WIRING:
 *   Red LED    -> GPIO 13 -> 220 ohm resistor -> GND
 *   Yellow LED -> GPIO 12 -> 220 ohm resistor -> GND
 *   Green LED  -> GPIO 14 -> 220 ohm resistor -> GND
 *
 * Sends state over Serial at 9600 baud so you can monitor in
 * Arduino Serial Monitor or with Python.
 */

#define RED_PIN    13
#define YELLOW_PIN 12
#define GREEN_PIN  14

#define GREEN_TIME   5000   // 5 seconds
#define YELLOW_TIME  2000   // 2 seconds
#define RED_TIME     5000   // 5 seconds

void setup() {
  Serial.begin(9600);
  pinMode(RED_PIN, OUTPUT);
  pinMode(YELLOW_PIN, OUTPUT);
  pinMode(GREEN_PIN, OUTPUT);

  // Quick startup test: blink all 3 once
  digitalWrite(RED_PIN, HIGH);
  digitalWrite(YELLOW_PIN, HIGH);
  digitalWrite(GREEN_PIN, HIGH);
  delay(500);
  digitalWrite(RED_PIN, LOW);
  digitalWrite(YELLOW_PIN, LOW);
  digitalWrite(GREEN_PIN, LOW);
  delay(500);

  Serial.println("Single traffic light test ready");
}

void setLight(const char* color) {
  digitalWrite(RED_PIN, LOW);
  digitalWrite(YELLOW_PIN, LOW);
  digitalWrite(GREEN_PIN, LOW);

  if (strcmp(color, "red") == 0)    digitalWrite(RED_PIN, HIGH);
  if (strcmp(color, "yellow") == 0) digitalWrite(YELLOW_PIN, HIGH);
  if (strcmp(color, "green") == 0)  digitalWrite(GREEN_PIN, HIGH);

  Serial.print("STATE:");
  Serial.println(color);
}

void loop() {
  setLight("red");
  delay(RED_TIME);

  setLight("green");
  delay(GREEN_TIME);

  setLight("yellow");
  delay(YELLOW_TIME);
}
