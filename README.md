# Face-Recognition
OpenCV Face Recognition

An image recognition model which works by detecting faces using haar_cascades and then recognizes them using LBPH algorithm (provided in opencv-contrib-python library)


Put all the face images you want your model to recognize in the `images` folder in the format as shown there.

### Install the dependencies

Run `pip install -r requiremnts.txt`

### Train your model and save train data
Run `python faces-train.py`

### Test your model against live webcam
Run `python faces.py`

### Output

![Screenshot](image.png)

