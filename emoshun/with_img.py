import cv2 as cv
from deepface import DeepFace
import matplotlib.pyplot as plt

imgPath = "./input_image.jpeg"

img = cv.imread(imgPath)
grey_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

face_classifier = cv.CascadeClassifier(
    cv.data.haarcascades + "haarcascade_frontalface_default.xml"
)

face = face_classifier.detectMultiScale(
    grey_img, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
)

for (x, y, w, h) in face:
    result = DeepFace.analyze(img_path=imgPath, actions=['emotion', 'race'], enforce_detection=False)
    emotion = result[0]['dominant_emotion']
    race = result[1]['dominant_race']

    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)
    # cv.putText(img, emotion, race, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv.putText(img, f"Emotion: {emotion}", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv.putText(img, f"Race: {race}", (x, y + h + 30), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# cv.imshow('Emoshun detection:')

img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

plt.figure(figsize=(20,10))
plt.imshow(img_rgb)
plt.axis('off')
plt.show()