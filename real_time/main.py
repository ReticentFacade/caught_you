import cv2 as cv

face_classifier = cv.CascadeClassifier(
    cv.data.haarcascades + "haarcascade_frontalface_default.xml"
)

video_capture = cv.VideoCapture(0)

def detect_bounding_box(vid):
    gray_img = cv.cvtColor(vid, cv.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(
        gray_img, 1.1, 5, minSize=(40,40)
    )

    for (x, y, w, h) in faces:
        cv.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
    return faces

while True:
    result, video_frame = video_capture.read() # read frames from the vid
    if result is False:
        break

    faces = detect_bounding_box(video_frame)
    cv.imshow("Face detection implementation", video_frame)

    if cv.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv.destroyAllWindows()