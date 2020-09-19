import face_recognition
from skimage.feature import hog
from skimage.transform import pyramid_gaussian
from skimage import color
from imutils.object_detection import non_max_suppression
import numpy as np
import cv2
import pickle


image = cv2.imread("Italo.jpeg")
img = cv2.resize(image, (300, 300))
cv2.imshow("", img)
cv2.waitKey()
filename = "Face_Detector.pkl"
model = pickle.load(open(filename, 'rb'))
filename = "Face_Recognizer_Model.pkl"
classifier = pickle.load(open(filename, 'rb'))

detections = []
(winW, winH) = (140, 140)
windowSize = (winW, winH)
downscale = 1.5


# define the sliding window:
def sliding_window(image_, stepSize, Size_Window):
    for val in range(0, image_.shape[0], stepSize):
        for val_ in range(0, image_.shape[1], stepSize):
            yield val_, val, image_[val: val + Size_Window[1], val_:val_ + Size_Window[0]]


def Image_Sliding(Image):
    scale = 0
    for resized in pyramid_gaussian(Image, downscale=1.5):
        for (x, y, window) in sliding_window(resized, stepSize=10, Size_Window=(winW, winH)):
            if window.shape[0] != winH or window.shape[1] != winW:
                continue
            if window.shape == (140, 140, 2):
                break
            window = color.rgb2gray(window)
            fds = hog(window, block_norm='L2-Hys')
            fds = list(fds)
            predictions = model.predict([fds])
            print(predictions)

            if predictions == 1:
                if model.decision_function([fds]) > 0.6:
                    print("Detection:: Location -> ({}, {})".format(x, y))
                    print("Scale ->  {} | Confidence Score {} \n".format(scale, model.decision_function([fds])))
                    detections.append((int(x * (downscale ** scale)),
                                       int(y * (downscale ** scale)), model.decision_function([fds]),
                                       int(windowSize[0] * (downscale ** scale)),
                                       int(windowSize[1] * (downscale ** scale))))
    rects = np.array([[x, y, x + w, y + h] for (x, y, _, w, h) in detections])
    rects_ = np.array([[x, y + h, x + w, y] for (x, y, _, w, h) in detections])
    sc = [score[0] for (x, y, score, w, h) in detections]
    print("detection confidence score: ", sc)
    sc = np.array(sc)
    pick = non_max_suppression(rects, probs=sc, overlapThresh=0.01)
    pick_ = non_max_suppression(rects_, probs=sc, overlapThresh=0.01)
    # if pick[0] is list:
    #     pick = pick[0]

    # rectangles = []

    try:
        # print(type(pick))
        # print(type(pick[0]))
        for (xA, yA, xB, yB) in pick:
            cv2.rectangle(Image, (xA, yA), (xB, yB), (0, 255, 0), 2)
        encodings = face_recognition.face_encodings(Image, pick_)
        print(encodings[0])
        prediction = classifier.predict([encoding.tolist() for encoding in encodings])
        prediction = prediction[0]
        print(prediction)
        cv2.putText(Image, text=prediction, org=(pick_[0][0], pick_[0][3]),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 0, 0), thickness=1)
        cv2.imshow("", Image)
        cv2.waitKey(0)
        return pick_
    except IndexError:
        print("wwe")


Image_Sliding(img)
