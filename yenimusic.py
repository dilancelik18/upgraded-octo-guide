import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
from picamera2 import Picamera2
import RPi.GPIO as GPIO
from time import sleep

# Buzzer ayarları
BUZZER_PIN = 4
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUZZER_PIN, GPIO.OUT)
buzzer = GPIO.PWM(BUZZER_PIN, 1000)  # 1 kHz PWM sinyali

# Notalar ve şarkılar
tones = {
    "E5": 659, "G5": 784, "A5": 880, "P": 0
}

happy_song = ["E5", "G5", "A5", "P", "E5", "G5", "B5", "A5", "P", "E5", "G5", "A5", "P", "G5", "E5"]
sad_song = ["E5", "E5", "E5", "P", "E5", "G5", "B5", "A5", "P", "E5", "G5", "A5", "P", "G5", "E5"]
angry_song = ["E5", "P", "G5", "P", "A5", "P", "E5", "P", "G5", "P", "B5", "P", "A5", "P", "E5", "P", "G5", "A5", "P", "G5", "E5"]

# Yüz tanıma modeli ve sınıflandırıcı
model = load_model('/home/pi/FER_model.h5')  # Model yolunu doğru ayarlayın
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Buzzer fonksiyonları
def playtone(frequency):
    if frequency != 0:
        buzzer.start(50)  # %50 duty cycle
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

        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image, predicted_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        return image, predicted_emotion
    return image, "No emotion detected"

# Kamera kurulumu
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration())
picam2.start()

# Gerçek zamanlı görüntü işleme
try:
    while True:
        frame = picam2.capture_array()
        image, predicted_emotion = get_emotion_from_image(frame)

        # Şarkı seçimi
        if predicted_emotion == 'Happy':
            song = happy_song
        elif predicted_emotion == 'Sad':
            song = sad_song
        elif predicted_emotion == 'Angry':
            song = angry_song
        else:
            song = []

        # Şarkıyı çalma
        if song:
            playsong(song)

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
