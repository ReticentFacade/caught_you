import cv2

imagePath = 'input_image.jpeg'

img = cv2.imread(imagePath) # Reads the image from specified filePath
# print(img.shape) # `img.shape` generates a/an vector/array that contains dimensions of that img

gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(gray_image.shape)

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

face = face_classifier.detectMultiScale(
    gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
)