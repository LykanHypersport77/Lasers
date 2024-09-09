int greenLightPin = 2;
int yellowLightPin = 3;
int redLightPin = 4;
int blueLightPin = 6;  // Fire mode light
int buttonPin = 5;     // Pin for fire mode button
int shootPin = 7;      // Pin to send 5V signal to shoot mechanism

bool fire_mode = false;          // Fire mode state
bool lastButtonState = HIGH;     // Store the last button state (HIGH = not pressed)
unsigned long lastFireTime = 0;  // Stores the time when the red light was last activated
const unsigned long fireCooldown = 10000;  // 10-second cooldown for firing

void setup() {
  pinMode(greenLightPin, OUTPUT);
  pinMode(yellowLightPin, OUTPUT);
  pinMode(redLightPin, OUTPUT);
  pinMode(blueLightPin, OUTPUT);
  pinMode(shootPin, OUTPUT);      // Output pin for firing mechanism
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
      digitalWrite(shootPin, LOW);        // Turn off the shooting mechanism
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

      unsigned long currentTime = millis();
      // Check if the cooldown period has passed
      if (fire_mode && (currentTime - lastFireTime >= fireCooldown)) {
        // If fire mode is active and cooldown period has passed, turn on the red light
        delay(500);  // Delay to allow tracking to stabilize (500ms)

        Serial.println("Fire mode is active, turning on red light and firing.");
        digitalWrite(redLightPin, HIGH);  // Red light ON (fire mode + tracking)
        digitalWrite(shootPin, HIGH);     // Output 5V signal to shoot mechanism
        delay(100);  // Fire for 100ms
        digitalWrite(shootPin, LOW);      // Turn off shooting signal after firing
        lastFireTime = currentTime;       // Reset the cooldown timer
      } else if (fire_mode && (currentTime - lastFireTime < fireCooldown)) {
        Serial.println("Fire cooldown in progress, cannot fire yet.");
        digitalWrite(redLightPin, LOW);   // Red light OFF during cooldown
      }
    }
  }
}
