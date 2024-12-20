import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
import matplotlib.pyplot as plt
import RPi.GPIO as GPIO  # Raspberry Pi GPIO kütüphanesi
from time import sleep

# Buzzer'ın GPIO4'e bağlı olduğunu varsayıyoruz
BUZZER_PIN = 4

# GPIO pinini ve PWM'i yapılandırıyoruz
GPIO.setmode(GPIO.BCM)  # BCM pin numaralandırma
GPIO.setup(BUZZER_PIN, GPIO.OUT)
buzzer = GPIO.PWM(BUZZER_PIN, 1000)  # 1 kHz frekansta PWM sinyali üret

# Tonlar ve şarkılar
tones = {
    "E5": 659, "G5": 784, "A5": 880, "P": 0,  # Örnek olarak sadece 3 nota yer verdim
    # Diğer tonlar...
}

Happy_song = ["E5", "G5", "A5", "P", "E5", "G5", "B5", "A5", "P", "E5", "G5", "A5", "P", "G5", "E5"]
sad_song = ["E5", "E5", "E5", "P", "E5", "G5", "B5", "A5", "P", "E5", "G5", "A5", "P", "G5", "E5"]
angry_song = ["E5", "P", "G5", "P", "A5", "P", "E5", "P", "G5", "P", "B5", "P", "A5", "P", "E5", "P", "G5", "A5", "P", "G5", "E5"]

# Yüz tanıma modeli
model = load_model('C:/Users/dilan.celik/Desktop/FER_model.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Duyguyu tespit eden fonksiyon
def get_emotion_from_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return None, None
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

# Kamera modülünden görüntü almak için OpenCV'yi kullanıyoruz
cap = cv2.VideoCapture(0)  # 0 numaralı cihazda kamera var (kamera modülü)

if not cap.isOpened():
    print("Kamera açılamadı!")
    exit()

# Buzzer ve şarkı çalma fonksiyonları
def playtone(frequency):
    if frequency != 0:
        buzzer.start(50)  # %50 duty cycle ile PWM başlat
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

# Gerçek zamanlı görüntü işleme ve duygu tespiti
while True:
    ret, frame = cap.read()
    if not ret:
        print("Kamera görüntüsü alınamadı!")
        break

    # Duyguyu tespit etme
    image, predicted_emotion = get_emotion_from_image(frame)

    # Hangi şarkıyı çalacağına karar verme
    if predicted_emotion == 'Happy':
        song = Happy_song
    elif predicted_emotion == 'Sad':
        song = sad_song
    elif predicted_emotion == 'Angry':
        song = angry_song
    else:
        song = []

    # Şarkıyı çalma
    playsong(song)

    # Yüzü ve duygu etiketini ekranda gösterme
    cv2.imshow("Emotion Detection", image)

    # 'q' tuşuna basılınca çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamera ve OpenCV kaynaklarını serbest bırakma
cap.release()
cv2.destroyAllWindows()

# Buzzer'ı kapatma
buzzer.stop()
GPIO.cleanup()
