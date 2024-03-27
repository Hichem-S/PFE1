#include "LiquidCrystal.h"
LiquidCrystal lcd(8,7,6,5,4,3);
int a = 0;
int b = 0;
const int trigPin = 12; // Broche TRIG du capteur ultrasonique connectée à la broche 9 de l'Arduino
const int echoPin = 2; // Broche ECHO du capteur ultrasonique connectée à la broche 10 de l'Arduino
// Pin numbers for the LEDs
const int redLedPin = 11;
const int yellowLedPin = 9;
const int greenLedPin = 10;

int sensorPin = 0;
const int soilPin = A1;
int soilValue = 1;
// Pin numbers
const int gasPin = A3; // Analog pin for gas sensor output
const int ledblue = 13; // Digital pin for LED
void setup()
{
  Serial.begin(9600);
  lcd.begin(16,2);
  // Initialize pins for ultrasonic sensor
  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);
  
  // Initialize pins for LEDs
  pinMode(redLedPin, OUTPUT);
  pinMode(yellowLedPin, OUTPUT);
  pinMode(greenLedPin, OUTPUT);
  pinMode(ledblue, OUTPUT); // Set LED pin as output
  
}
 
void loop()
{
 // capteur de temperature 
 int reading = analogRead(sensorPin);

 // measure the 5v with a meter for an accurate value
 //In particular if your Arduino is USB powered

 float voltage = reading * 4.68;
 voltage /= 1024.0;
 
 // now print out the temperature

 float temperatureC = (voltage - 0.5) * 100;
 Serial.print(temperatureC);
 Serial.println(" degrees C ");

   lcd.setCursor(0,0);
   lcd.print("Temperature Value ");
   lcd.setCursor(0,1);
   lcd.print(" degrees C");
   lcd.setCursor(11,1);
   lcd.print(temperatureC);
 
 delay(100);
  //capteur d'humidité du sol
  // Read the soil moisture sensor value
  soilValue = analogRead(soilPin);
  // Print the soil moisture sensor value to the serial monitor
  Serial.print("Soil Moisture: ");
  Serial.println(soilValue);

  delay(1000); // Delay for 1 second
  // pir sensor 
  a = analogRead(A2);
  b = map (a,0,1023,0,255);
    Serial.println ("Motion detected");

  if (b>100)
  {
  Serial.println (b);
    delay(100);}
  // ultrason sensor 
        long duration, distance;
  
  // Clear the trigger pin
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  
  // Send a 10 microsecond pulse to the trigger pin
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);
  
  // Measure the time taken by the echo pin to go high
  duration = pulseIn(echoPin, HIGH); // bech tguiselna signal ki yabda echo pin yekhdem
  
  // Calculate the distance in cm
  distance = duration * 0.034 / 2; // pour convertir la durée mesurée par le capteur ultrasonique en une distance en centimètres
  
  // Print the distance to the serial monitor
  Serial.print("Distance: ");
  Serial.print(distance);
  Serial.println(" cm");
  
  // Turn on the appropriate LED based on the distance
  if (distance < 20) {
    digitalWrite(redLedPin, HIGH);
    digitalWrite(yellowLedPin, LOW);
    digitalWrite(greenLedPin, LOW);
  } else if (distance < 50) {
    digitalWrite(redLedPin, LOW);
    digitalWrite(yellowLedPin, HIGH);
    digitalWrite(greenLedPin, LOW);
  } else {
    digitalWrite(redLedPin, LOW);
    digitalWrite(yellowLedPin, LOW);
    digitalWrite(greenLedPin, HIGH);
  }
  
  // Delay for a short time before next reading
  delay(100);
  // Gaz sensor
  int gasValue = analogRead(gasPin); // Read gas sensor value

  // Print the gas sensor value to the serial monitor
  Serial.print("Gas Value: ");
  Serial.println(gasValue);

  // If gas value is above a certain threshold, turn on LED
  if (gasValue > 500) {
    digitalWrite(ledblue, HIGH); // Turn on LED
    Serial.println("Gas detected! LED on.");
  } else {
    digitalWrite(ledblue, LOW); // Turn off LED
  }

  delay(1000); // Delay for 1 second
}