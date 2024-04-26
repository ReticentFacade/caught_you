# Real-time facial recognition

- Since this is real-time, obviously no need to load any image. Directly start with:

### Step 1: Pre-requisites

```py
import cv2 as cv

face_classifier = cv.CascadeClassifier(
    cv.data.haarcascades + "haarcascade_frontalface_default.xml"
)
```

### Step 2: Access the webcam

```py
video_capture = cv.VideoCapture(0)
```

- The `0` parameter tekls `OpenCV` to use the default camera on the device.

### Step 3: Identifying Faces in the Video Stream

- Same thing as the one in [with_images](../with_images/notes.md) but in a function here.

### Step 4: Creating a Loop for Real-Time Face Detection

```py
video_capture.read()
```

- `read()`: reads frames from the vid (video input)

```py
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
```

- Created a loop that continuously reads frames from a video using the `video_capture.read()` function.
- If the frame is not read successfully, the loop is terminated using the `break` statement.
- The detect_bounding_box() function is then applied to each frame to detect faces.
- The resulting frames are displayed in a window titled "Face detection implementation" using the `cv2.imshow()` function.
- The loop is terminated if the user presses the "q" key using the cv2.waitKey() function.
- Finally, the `video_capture.release()` function is called to release the video capture object and the `cv2.destroyAllWindows()` function is called to close all windows.
- Nutshell: This code is used to detect faces in a **video stream and display the processed frames in a window.**
