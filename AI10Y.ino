//-- Slider Pins --
const int potPin1 = A0;
const int potPin2 = A1;

//-- Image Select Pins --
const int imgPin1 = 2;
const int imgPin2 = 3;
const int imgPin3 = 4;
const int imgPin4 = 5;

//-- Photo Take Pins --
const int capturePin = 8;
const int retakePin = 9;
const int acceptPin = 10;

//-- Section Pins --
const int imgSectionPin = 6;
const int sliderSectionPin = 7;
const int photoSectionPin = 11;
const int generateSectionPin = 12;

//-- Global Variables --
int selectedImg = 5;
int sectionPart = 0;
float potValue1;
float potValue2;
bool captureImg = false;
bool acceptImg = false;

#define ledGroup1 14
#define ledGroup2 15
#define ledGroup3 16
#define ledGroup4 17
#define ledGroup5 18
#define ledGroup6 19

//-- Setup --
void setup() {
  Serial.begin(115200); // Initialize serial communication at 115200 bps

  // Configuring buttons
  pinMode(imgPin1, INPUT_PULLUP);
  pinMode(imgPin2, INPUT_PULLUP);
  pinMode(imgPin3, INPUT_PULLUP);
  pinMode(imgPin4, INPUT_PULLUP);
  pinMode(capturePin, INPUT_PULLUP);
  pinMode(retakePin, INPUT_PULLUP);
  pinMode(acceptPin, INPUT_PULLUP);
  pinMode(imgSectionPin, INPUT_PULLUP);
  pinMode(sliderSectionPin, INPUT_PULLUP);
  pinMode(photoSectionPin, INPUT_PULLUP);
  pinMode(generateSectionPin, INPUT_PULLUP);

  pinMode(ledGroup1, OUTPUT);   
  pinMode(ledGroup2, OUTPUT);   
  pinMode(ledGroup3, OUTPUT);  
  pinMode(ledGroup4, OUTPUT);    
  pinMode(ledGroup5, OUTPUT);   
  pinMode(ledGroup6, OUTPUT);   
}

void imgSelect() {
  digitalWrite(ledGroup1, HIGH);
  if (digitalRead(imgPin1) == LOW) {
    selectedImg = 1;
  } 
  else if (digitalRead(imgPin2) == LOW) {
    selectedImg = 2;
  } 
  else if (digitalRead(imgPin3) == LOW) {
    selectedImg = 3;
  } 
  else if (digitalRead(imgPin4) == LOW) {
    selectedImg = 4;
  }
}

void sliderSelect() {
  // Read analog values
  digitalWrite(ledGroup3, HIGH);
  int rawPotValue1 = analogRead(potPin1);
  int rawPotValue2 = analogRead(potPin2);

  // Map values from range 0-1023 to 0-1000
  int mappedPotValue1 = map(rawPotValue1, 0, 1023, 0, 1000);
  int mappedPotValue2 = map(rawPotValue2, 0, 1023, 0, 1000);

  // Convert to float and scale to 0.00 - 1.00
  potValue1 = float(mappedPotValue1) / 1000.0;
  potValue2 = float(mappedPotValue2) / 1000.0;
}

void photoSelect() {
  digitalWrite(ledGroup4, HIGH);
  if (digitalRead(capturePin) == LOW && !captureImg) {
    Serial.println("2,c");
    captureImg = true;
  }
  else if (digitalRead(retakePin) == LOW && captureImg) {
    Serial.println("2,r");
    captureImg = false;
    acceptImg = false;
    digitalWrite(ledGroup5, LOW);
  }
  else if (digitalRead(acceptPin) == LOW && captureImg && !acceptImg) {
    Serial.println("2,a");
    acceptImg = true;
  }
}

void loop() {
  if (digitalRead(imgSectionPin) == LOW && sectionPart == 0 && selectedImg != 5) {
    sectionPart = 1;
    Serial.print("0,");
    Serial.println(selectedImg);
  }
  else if (digitalRead(sliderSectionPin) == LOW && sectionPart == 1) {
    sectionPart = 2;
    Serial.print("1,");
    Serial.print(potValue1);
    Serial.print(",");
    Serial.println(potValue2);
  }
  else if (digitalRead(photoSectionPin) == LOW && sectionPart == 2 && captureImg == true && acceptImg == true) {
    sectionPart = 3;
    Serial.println("2,l");
  }
  else if (digitalRead(generateSectionPin) == LOW && sectionPart == 3) {
    sectionPart = 0;
    captureImg = false;
    acceptImg = false;
    selectedImg = 5;
    Serial.println("3,g");
    digitalWrite(ledGroup1, LOW);
    digitalWrite(ledGroup2, LOW);
    digitalWrite(ledGroup3, LOW);
    digitalWrite(ledGroup4, LOW);
    digitalWrite(ledGroup5, LOW);
    digitalWrite(ledGroup6, LOW);
    delay(5000);
    digitalWrite(ledGroup1, HIGH);
  }

  switch (sectionPart) {
    case 0:
      imgSelect();
      break;
    case 1:
      sliderSelect();
      break;
    case 2:
      photoSelect();
      break;
    case 3:
      digitalWrite(ledGroup6, HIGH);
      break;
  }

  if (selectedImg != 5){
    digitalWrite(ledGroup2, HIGH);
  } 

  if (acceptImg == true && captureImg == true){
    digitalWrite(ledGroup5, HIGH);
  }
}
