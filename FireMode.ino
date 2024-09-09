int greenLightPin = 2;
int yellowLightPin = 3;
int redLightPin = 4;
int blueLightPin = 6;  // Fire mode light
int buttonPin = 5;     // Pin for fire mode button

bool fire_mode = false;  // Fire mode state
bool lastButtonState = HIGH;  // Store the last button state (HIGH = not pressed)

void setup() {
  pinMode(greenLightPin, OUTPUT);
  pinMode(yellowLightPin, OUTPUT);
  pinMode(redLightPin, OUTPUT);
  pinMode(blueLightPin, OUTPUT);
  pinMode(buttonPin, INPUT_PULLUP);  // Use internal pull-up for button pin

  Serial.begin(115200);  // Initialize serial communication
  Serial.println("Arduino ready");
}

void loop() {
  // Read the current state of the button
  int buttonState = digitalRead(buttonPin);

  // Toggle fire mode when button is pressed (change from HIGH to LOW)
  if (buttonState == LOW && lastButtonState == HIGH) {
    fire_mode = !fire_mode;  // Toggle fire mode
    Serial.print("Fire mode toggled: ");
    Serial.println(fire_mode ? "ON" : "OFF");

    // Control the blue light based on fire mode
    digitalWrite(blueLightPin, fire_mode ? HIGH : LOW);  // Fire mode ON or OFF

    delay(100);  // Debounce delay to prevent multiple toggles from one press
  }

  // Update last button state
  lastButtonState = buttonState;

  // Handle serial commands for tracking
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    Serial.print("Command received: ");
    Serial.println(command);

    if (command == "idle") {
      // No drone detected
      Serial.println("No drone detected. Idle mode.");
      digitalWrite(greenLightPin, HIGH);  // Green light ON (idle)
      digitalWrite(yellowLightPin, LOW);  // Yellow light OFF (not tracking)
      digitalWrite(redLightPin, LOW);     // Red light OFF (fire mode inactive)
    } else if (command == "tracking") {
      // Drone detected and tracking
      Serial.println("Drone detected. Tracking mode.");
      digitalWrite(greenLightPin, LOW);   // Green light OFF (drone detected)
      digitalWrite(yellowLightPin, HIGH); // Yellow light ON (tracking)
      digitalWrite(redLightPin, LOW);     // Red light OFF (only active in fire mode)
    } else if (command == "fire_tracking") {
      // Fire mode is on and a drone is being tracked
      Serial.println("Drone detected and fire mode is ON. Fire tracking mode.");
      digitalWrite(greenLightPin, LOW);   // Green light OFF (drone detected)
      digitalWrite(yellowLightPin, HIGH); // Yellow light ON (tracking)

      // If fire mode is active, turn on the red light
      if (fire_mode) {
        Serial.println("Fire mode is active, turning on red light.");
        digitalWrite(redLightPin, HIGH);  // Red light ON (fire mode + tracking)
      } else {
        delay(100);// added so that the system fires 100ms after detecting drone in fire mode...hopefully
        Serial.println("Fire mode is OFF, keeping red light off.");
        digitalWrite(redLightPin, LOW);   // Red light OFF (no fire mode)
      }
    }
  }
}
