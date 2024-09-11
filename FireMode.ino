#include <LiquidCrystal.h>

// Initialize the LCD with pin assignments
LiquidCrystal lcd(8, 9, 10, 11, 12, 13);  // RS, EN, D4, D5, D6, D7

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
  // Initialize LCD
  lcd.begin(16, 2);  // Set up the 16x2 LCD
  lcd.print("System Ready");  // Print initial message
  delay(2000);  // Display for 2 seconds
  lcd.clear();  // Clear the screen after 2 seconds

  // Initialize LED and button pins
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

    delay(200);  // Debounce delay to prevent multiple toggles from one press
  }

  // Update last button state
  lastButtonState = buttonState;

  // Handle serial commands for tracking
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    Serial.print("Command received: "); 
    Serial.println(command);

    // Clear LCD before displaying new information
    lcd.clear();

    if (command == "idle") {
      // No drone detected
      Serial.println("No drone detected. Idle mode.");
      lcd.print("Idle: No Drone");        // Update LCD
      digitalWrite(greenLightPin, HIGH);  // Green light ON (idle)
      digitalWrite(yellowLightPin, LOW);  // Yellow light OFF (not tracking)
      digitalWrite(redLightPin, LOW);     // Red light OFF (fire mode inactive)
      digitalWrite(shootPin, LOW);        // Turn off the shooting mechanism
    } else if (command == "tracking") {
      // Drone detected and tracking
      Serial.println("Drone detected. Tracking mode.");
      lcd.print("Tracking Drone");        // Update LCD
      digitalWrite(greenLightPin, LOW);   // Green light OFF (drone detected)
      digitalWrite(yellowLightPin, HIGH); // Yellow light ON (tracking)
      digitalWrite(redLightPin, LOW);     // Red light OFF (only active in fire mode)
    } else if (command == "fire_tracking") {
      // Fire mode is on and a drone is being tracked
      Serial.println("Drone detected and fire mode is ON. Fire tracking mode.");
      lcd.setCursor(0, 0);
      lcd.print("Firing at Drone");       // Update LCD on first line

      unsigned long currentTime = millis();
      unsigned long timeUntilFire = (fireCooldown - (currentTime - lastFireTime)) / 1000;  // Time until next fire in seconds

      // Check if the cooldown period has passed
      if (fire_mode && (currentTime - lastFireTime >= fireCooldown)) {
        // Fire mode is active and cooldown period has passed
        Serial.println("Fire mode is active, firing now.");
        digitalWrite(redLightPin, HIGH);  // Red light ON (ready to fire)
        digitalWrite(shootPin, HIGH);     // Output 5V signal to shoot mechanism
        delay(100);  // Fire for 100ms
        digitalWrite(shootPin, LOW);      // Turn off shooting signal after firing
        lastFireTime = currentTime;       // Reset the cooldown timer
      } else if (fire_mode && (currentTime - lastFireTime < fireCooldown)) {
        // Fire mode is active, but the system is cooling down
        lcd.setCursor(0, 1);  // Move to second line
        lcd.print("Cooldown: ");
        lcd.print(timeUntilFire);  // Print remaining cooldown time
        lcd.print("s");

        Serial.println("Cooldown active, cannot fire yet.");
        digitalWrite(redLightPin, LOW);   // Red light OFF during cooldown
      }

      // During tracking in fire mode, yellow light should still be ON
      digitalWrite(greenLightPin, LOW);   // Green light OFF (not idle)
      digitalWrite(yellowLightPin, HIGH); // Yellow light ON (tracking)
    }
  }
}
