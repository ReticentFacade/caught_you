from deepface import DeepFace
import cv2 as cv

face_classifier = cv.CascadeClassifier(
    cv.data.haarcascades + "haarcascade_frontalface_default.xml"
)

log_file = open("./ethnicity_log.txt", "a")
print("log_file opened...")
video_capture = cv.VideoCapture(0)

def detect_border(vid):
    print("detect_border function started...")
    gray_img = cv.cvtColor(vid, cv.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(
        gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )

    for (x, y, w, h) in faces: 
        print("inside the for loop now...")
        cv.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 5)

        face = gray_img[y:y + h, x:x + w]
        # try:
        print("trying now")
        result = DeepFace.analyze(face, actions=['race'])
        ethnicity = result['dominant_ethnicity']
        cv.putText(vid, ethnicity, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        print("writing in log_file...")
        log_file.write(f"Ethnicity: {ethnicity}\n")

        # except: 
            # pass

    return faces

while True: 
    result, video_frame = video_capture.read()
    if result is False:
        break

    faces = detect_border(video_frame)
    cv.imshow("App", video_frame)

    if (cv.waitKey(1) & 0xFF == ord("q")):
        break

log_file.close()
print("log_file closed")
video_capture.release()
cv.destroyAllWindows()