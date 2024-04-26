import cv2
import matplotlib.pyplot as plt

imagePath = '../input_image.jpeg'

img = cv2.imread(imagePath) # Reads the image from specified filePath
# print(img.shape) # `img.shape` generates a/an vector/array that contains dimensions of that img

gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# print(gray_image.shape)

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
# print(f"Face classifier:\n", face_classifier)

face = face_classifier.detectMultiScale(
    gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
)

# print("Face:\n", face)
# print(face)

for (x, y, w, h) in face: 
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# print(img_rgb)

plt.figure(figsize=(20,10))
plt.imshow(img_rgb)
plt.axis('off')
plt.show()