import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
from picamera2 import Picamera2
import RPi.GPIO as GPIO
from time import sleep

# Buzzer ayarları
BUZZER_PIN = 17
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUZZER_PIN, GPIO.OUT)
buzzer = GPIO.PWM(BUZZER_PIN, 1000)  # 1 kHz PWM sinyali

# Notalar ve şarkılar
tones = {
    "E5": 659, "G5": 784, "A5": 880, "B5": 987, "P": 0
}

songs = {
    "Happy": ["E5", "G5", "A5", "P", "E5", "G5", "B5", "A5", "P", "E5", "G5", "A5", "P", "G5", "E5"],
    "Sad": ["C4", "D4", "E4", "F4", "P", "F4", "E4", "D4", "C4"],
    "Angry": ["C5", "B4", "A4", "G4", "F4", "E4", "P", "C5"]
}

# Yüz tanıma modeli ve sınıflandırıcı
model = load_model('/home/pi/FER_model.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Buzzer fonksiyonları
def playtone(frequency):
    if frequency != 0:
        buzzer.start(50)
        buzzer.ChangeFrequency(frequency)
    else:
        buzzer.stop()

def bequiet():
    buzzer.stop()

def playsong(mysong):
    for note in mysong:
        if note == "P":
            bequiet()
        else:
            playtone(tones[note])
        sleep(0.2)
    bequiet()

# Duygu tespiti fonksiyonu
def get_emotion_from_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return image, "No face detected"

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = face.astype('float32') / 255
        face = np.expand_dims(face, axis=-1)
        face = np.expand_dims(face, axis=0)

        emotion_predictions = model.predict(face)
        max_index = np.argmax(emotion_predictions[0])
        predicted_emotion = emotion_labels[max_index]

        # Çerçeve ve metin çizimi
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image, predicted_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        return image, predicted_emotion
    return image, "No emotion detected"

# Kamera kurulumu
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration())
picam2.start()

# Gerçek zamanlı görüntü işleme
last_emotion = None  # Son algılanan duygu

try:
    while True:
        frame = picam2.capture_array()
        image, predicted_emotion = get_emotion_from_image(frame)

        # Yeni bir duygu algılandığında şarkıyı değiştir
        if predicted_emotion != last_emotion:
            print(f"Duygu değişti: {predicted_emotion}")
            if predicted_emotion in songs:
                playsong(songs[predicted_emotion])
            else:
                print("Tanımlı olmayan bir duygu algılandı.")
            last_emotion = predicted_emotion  # Yeni duyguyu kaydet

        # Görüntüyü ekranda göster
        cv2.imshow("Emotion Detection", image)

        # Çıkış için 'q' tuşu
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Çıkış yapılıyor...")

finally:
    # Kaynakları temizle
    picam2.close()
    buzzer.stop()
    GPIO.cleanup()
    cv2.destroyAllWindows()
