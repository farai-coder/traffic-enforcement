/*
 * Traffic Light Controller - 4 Mini Traffic Lights
 * =================================================
 * Controls 4 traffic lights at a model intersection.
 *
 * Layout (looking from above):
 *
 *          TL2 (South-facing)
 *            |
 *   TL3 ----+---- TL1 (East-facing)
 *   (West)   |
 *          TL4 (North-facing)
 *
 * TL1 & TL3 (East-West) run together
 * TL2 & TL4 (North-South) run together
 *
 * Also sends current light state over Serial so the
 * Python detection system knows the light state.
 *
 * Serial output format:
 *   STATE:ew:green,ns:red
 * ew = East-West pair (TL1 + TL3), ns = North-South pair (TL2 + TL4)
 * Legacy single-line STATE:green still supported (treated as ew by Python)
 *
 * WIRING (ESP32 GPIO pins):
 * -------------------------
 * Traffic Light 1 (East-facing):
 *   Red    -> GPIO 13
 *   Yellow -> GPIO 12
 *   Green  -> GPIO 14
 *
 * Traffic Light 2 (South-facing):
 *   Red    -> GPIO 27
 *   Yellow -> GPIO 26
 *   Green  -> GPIO 25
 *
 * Traffic Light 3 (West-facing):
 *   Red    -> GPIO 33
 *   Yellow -> GPIO 32
 *   Green  -> GPIO 23
 *
 * Traffic Light 4 (North-facing):
 *   Red    -> GPIO 19
 *   Yellow -> GPIO 18
 *   Green  -> GPIO 5
 *
 * All LED ground wires -> ESP32 GND
 * Each LED needs a 220 ohm resistor in series
 */

// ---- Traffic Light 1 (East) ----
#define TL1_RED    13
#define TL1_YELLOW 12
#define TL1_GREEN  14

// ---- Traffic Light 2 (South) ----
#define TL2_RED    27
#define TL2_YELLOW 26
#define TL2_GREEN  25

// ---- Traffic Light 3 (West) ----
#define TL3_RED    33
#define TL3_YELLOW 32
#define TL3_GREEN  23

// ---- Traffic Light 4 (North) ----
#define TL4_RED    19
#define TL4_YELLOW 18
#define TL4_GREEN   5

// Timing (milliseconds)
#define GREEN_TIME   10000  // 10 seconds green
#define YELLOW_TIME   3000  // 3 seconds yellow
#define RED_CLEAR_TIME 4000 // 4 seconds all-red clearance (time to move vehicle)

// All pins in arrays for easy setup
const int allPins[] = {
  TL1_RED, TL1_YELLOW, TL1_GREEN,
  TL2_RED, TL2_YELLOW, TL2_GREEN,
  TL3_RED, TL3_YELLOW, TL3_GREEN,
  TL4_RED, TL4_YELLOW, TL4_GREEN
};

void setup() {
  Serial.begin(9600);

  // Set all pins as output
  for (int i = 0; i < 12; i++) {
    pinMode(allPins[i], OUTPUT);
    digitalWrite(allPins[i], LOW);
  }

  Serial.println("Traffic Light Controller Ready");
}

// Helper: turn off all LEDs on a traffic light
void allOff(int red, int yellow, int green) {
  digitalWrite(red, LOW);
  digitalWrite(yellow, LOW);
  digitalWrite(green, LOW);
}

// Helper: set a traffic light to a specific state
void setLight(int red, int yellow, int green, char state) {
  allOff(red, yellow, green);
  switch (state) {
    case 'R': digitalWrite(red, HIGH); break;
    case 'Y': digitalWrite(yellow, HIGH); break;
    case 'G': digitalWrite(green, HIGH); break;
  }
}

// Set East-West pair (TL1 + TL3)
void setEastWest(char state) {
  setLight(TL1_RED, TL1_YELLOW, TL1_GREEN, state);
  setLight(TL3_RED, TL3_YELLOW, TL3_GREEN, state);
}

// Set North-South pair (TL2 + TL4)
void setNorthSouth(char state) {
  setLight(TL2_RED, TL2_YELLOW, TL2_GREEN, state);
  setLight(TL4_RED, TL4_YELLOW, TL4_GREEN, state);
}

void sendState(const char* ewState, const char* nsState) {
  Serial.print("STATE:ew:");
  Serial.print(ewState);
  Serial.print(",ns:");
  Serial.println(nsState);
}

void loop() {
  // --- Phase 1: East-West GREEN, North-South RED ---
  setEastWest('G');
  setNorthSouth('R');
  sendState("green", "red");
  delay(GREEN_TIME);

  // --- Phase 2: East-West YELLOW, North-South RED ---
  setEastWest('Y');
  setNorthSouth('R');
  sendState("yellow", "red");
  delay(YELLOW_TIME);

  // --- Phase 3: All RED (clearance) ---
  setEastWest('R');
  setNorthSouth('R');
  sendState("red", "red");
  delay(RED_CLEAR_TIME);

  // --- Phase 4: North-South GREEN, East-West RED ---
  setNorthSouth('G');
  setEastWest('R');
  sendState("red", "green");
  delay(GREEN_TIME);

  // --- Phase 5: North-South YELLOW, East-West RED ---
  setNorthSouth('Y');
  setEastWest('R');
  sendState("red", "yellow");
  delay(YELLOW_TIME);

  // --- Phase 6: All RED (clearance) ---
  setEastWest('R');
  setNorthSouth('R');
  sendState("red", "red");
  delay(RED_CLEAR_TIME);
}
