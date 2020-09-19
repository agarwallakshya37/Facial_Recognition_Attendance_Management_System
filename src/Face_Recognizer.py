from Testing_Face_detector import Image_Sliding
import os
import openface
import pandas as pd
from sklearn.svm import SVC
import face_recognition
import cv2
import pickle

#
path = "../Images"
face_encodings = []
Names = []
face_aligner = openface.AlignDlib("shape_predictor_68_face_landmarks.dat")


def encoding_images():
    for root, directory, filenames in os.walk(path):
        for file_ in filenames:
            path_image = os.path.join(".", root, file_)
            # print(path_image)
            image = cv2.imread(path_image)
            image_new = cv2.resize(image, (300, 300))
            image_ = cv2.resize(image, (300, 300))
            # cv2.imshow("", image)
            # cv2.waitKey()
            name = file_.split('.')
            name__ = ""
            for n in name:
                if n == '.' or n == 'jpeg' or n == 'jpg' or n == 'png' or n == 'PNG':
                    pass
                else:
                    name__ += n
            rectangles = Image_Sliding(image_new)
            print(rectangles[0])
            print(rectangles[0][0], rectangles[0][3])
            try:
                encodings = face_recognition.face_encodings(image_, rectangles)
                face_encodings.append(encodings[0].tolist())
                Names.append(name__)
                cv2.imshow("", image_new)
                cv2.waitKey()
            except:
                print("Error")
            # print(len(encodings))
            # face_encodings.append(encodings[0].tolist())
            # print(name__)
            # Names.append(name__)
            # print(encodings)


encoding_images()

cols = []
for i in face_encodings:
    for j in range(0, len(i), 1):
        cols.append("Encoded_Value - {}".format(j + 1))
    break
print(cols)
image_encoding_dataset = pd.DataFrame(face_encodings, columns=cols)
image_encoding_dataset["Names"] = Names
image_encoding_dataset.to_csv("Encoded Images And Test Data Set.csv")

dataset = pd.read_csv("Encoded Images And Test Data Set.csv")
X = dataset.iloc[:, 1: 129].values
y = dataset.iloc[:, 129].values
classifier = SVC(kernel='linear')
classifier.fit(X, y)

Pkl_Filename = "Face_Recognizer_Model.pkl"
with open(Pkl_Filename, 'wb') as file:
    pickle.dump(classifier, file)
