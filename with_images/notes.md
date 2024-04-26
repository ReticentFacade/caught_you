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

The above thing _BASICALLY MEANS,_
The file haarcascade_frontalface_default.xml is a pre-trained classifier provided by OpenCV for detecting faces.

The overall effect of this code is to create a face_classifier object that can be used to detect faces in images. This is done by loading the Haar cascade classifier for frontal face detection.
Once the classifier is loaded, it can be used to detect faces in images by calling the detectMultiScale method on the classifier object, passing in the image to be analyzed.

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
- `minNeighbors`: To understand it, read the section below first -->

---

- `Classifier` is responsible for categorising what input data is relevant to the object we're trying to detect (i.e., which input data is `positive` and which is `negative`).

- A `Classifier` can use data structures like: `Decision Tree`, `Support Vector Machines (SVMs)`, `Neural Networks`, `K-Nearest Neighbours (KNN)`.
- Then, may use `ensemble methods` like: `Random Forests`, `Gradient boosting machines`.
- **What does the DS of a `classfier` depend on?** It depends on the type of algo you're using.

- In `Haar Cascade`, classifiers typically use simple DS' like `Decision Tree` or `Boosted Decision Tree`.

---

...back to the topic now -->

- `minNeighbors`: Right. The `cascade classifier` applies a `sliding window` through the img to detect faces in it.
- Initially, the classifier captures a large no. of FALSE POSITIVES and `minNeighbors` **parameter is responsible for eliminating them.**

HOW????? <---- FIND OUT.

- `minSize`: Sets minimum size of the object to be detected. The model will ignore faces that are smaller than the `minSize` specified.

### Step 6: Drawing a Bounding Box [i.e., Border]

```py
for (x, y, w, h) in face:
  cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)
```

BGR(0, 255, 0) is obviously Green [LIME, speficially].
The `4` at the end is the thickness of the bounding box (basically, border).

### Step 7: Displaying the Image

- To display the img, first convert `BGR` back to `RGB`.

```py
img_rbg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
```

- And at the end use `Matplotlib` library to display the img

`py`

---

## Bird's eye view of `Haar Cascade`

`Haar Cascade` is just a fancy name for a pattern recognition algo.

**Mechanism of this algo:**

- Start by selecting a set of `positive` and `negative` "training images" containing the object we want to detect _and_ background images.
  { `Positive Image` = Contains the object we want to detect; `Negative Image` = Doesn't contain that object } --> This step is called `Initialisation`
- Extract Haar-like features from these imgs (capturing patterns & structures) --> This step is called `Feature Extraction`
- `Training`: Use `Adaboost algo` (?) to select & combine the most "discriminative" features into a strong classifier
- Organize the strong classifier into a cascade of stages, each consisting of multiple weak classifiers --> `Cascade Construction`
- During the detection phase, apply the cascade of classifiers to sliding windows across the image. At each stage, if a region passes all the weak classifiers, it proceeds to the next stage. If it fails at any stage, it's quickly rejected, improving computational efficiency --> `Detection`
