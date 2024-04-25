# Creating an image recognition tool

**Aim:** Learn mechanism, implementation/application of `OpenCV`

## 1.1 OpenCV functions in `main.py`

### Step 1: Import the OpenCV Package

- `imread`: OpenCV's function to read images.

_Mechanism of `imread`?_

- It will load the image from the specified filePath and return it in the form of a `Numpy array`.

### Step 2: Read the Image

- `shape`: Prints the dimensions of the aforementioned array.

When we ran `img.shape`, it generated a 3D array. What does the array contain?

```py
import cv2
imagePath = 'input_image.jpeg'
img = cv2.imread(imagePath)

print(img.shape)
```

```sh
(caught_you) ➜  caught_you python3 main.py
(4181, 2787, 3)
```

The array's values represent the picture's: 1. `Height` 2. `Width` 3. `Channels`

- Three channels were used to depict this coloured input_image: Blue, Green and Red

---

**Note:** While the conventional sequence used to represent images is `RGP`, the OpenCV library USES THE OPPOSITE LAYOUT: Blue, Green, Red.

_"RGB is commonly used in image editing and display applications, where the order is assumed as red, green, and blue. On the other hand, BGR is often used in image processing applications, and the order is assumed blue, green, and red."_

---

### Step 3: Convert the Image to Grayscale

- **Why are we doing this?** To improve computational efficiency.
- **Why do grayscale images have more compuational efficiency?**
  Gray-scale images require less memory and computational resources to process, making them faster and more efficient for training.
  Gray-scaling helps in simplifying algorithms and as well eliminates the complexities related to computational requirements.
  This is because grayscale compressors an image to its barest minimum pixel.

- **How do we convert an image from one colour to another?**

```py
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(gray_image.shape)
```

Where,

- `cv2.cvtColor` is a method to convert an image from one `"colour space"` to another. [There are more than 150 color-space conversion methods available in OpenCV]
- `cv2.COLOR_BGR2GRAY` is the color conversion code

Produces:

```sh
(caught_you) ➜  caught_you python3 main.py
(4181, 2787)
```

Note that this array has only two values since the image is now grayscale and NO LONGER HAS THE THIRD COLOR CHANNEL.

### Step 4: Load the Classifier

`Haar cascades` is a (built into OpenCV) pre-trained classifier.
Basically, detects objects in an image (like face, smile, eyes etc.)
It uses a combination of `weak classifiers` to produce `strong classifiers`

```py
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
```

- We used a file called `haarcascade_frontalface_default.xml`. This **classifier** is **designed specifically for detecting frontal faces** in visual input.

### Step 5: Perform the Face Detection

We'll perform face detection on the grayscale img.

```py
face = face_classifier.detectMultiScale(
    gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
)
```

Where,

- `detectMultiScale()`: method used to identify faces of different sizes in the input image.
- `scaleFactor`: parameter that `scales down` the size of the input image so that it's easier for the algo to detect larger faces. Having a `scaleFactor` of `1.1` means reducing the img_size by `10%`.
