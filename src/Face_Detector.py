import os
import cv2
from sklearn.svm import SVC
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle
import dlib
import openface


data = []
labels = []
path_faces = r"..\Img"
path_back = r"..\Background"


# print(os.name)
# print(os.getcwd())
# print(os.path.abspath('.'))
# print(os.listdir('.'))

cols = []
face_detector = dlib.get_frontal_face_detector()
face_pose_detector = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_aligner = openface.AlignDlib("shape_predictor_68_face_landmarks.dat")


def Required_Features(images, name):
    detected_faces = face_detector(images, 0)
    for i, face_ in enumerate(detected_faces):
        alignedFaces = face_aligner.align(140, images, face_, landmarkIndices=[openface.AlignDlib.OUTER_EYES_AND_NOSE])
        cv2.imwrite("Aligned_Faces_1/{}.jpg".format(name), alignedFaces)
        return alignedFaces


def Data_Preparation_For_Photos_With_Face():
    data_ = []
    label = []
    ind = 0
    for root, directory, filenames in os.walk(path_faces):
        # print(root, "----", directory, "----", filenames)

        for dir_ in directory:
            dire = os.path.basename(root)
            # print(dir_)
            path_image = os.path.join("..", dire, dir_)
            # print(path_image)
            for root_, directory_, filenames_ in os.walk(path_image):
                # print(root_, "===", directory_, "===", filenames_)
                image_no = 0

                for file in filenames_:
                    dir_image = os.path.basename(root_)
                    image_no += 1

                for file in filenames_:
                    dir_image = os.path.basename(root_)
                    # print(dir_image)
                    path_image = os.path.join(root, dir_image, file)
                    # print(path_image)
                    image = cv2.imread(path_image)
                    # cv2.imshow("", image_)
                    # cv2.waitKey()
                    # image_ = cv2.resize(image_, (400, 400))
                    name = file.split('.')
                    name = name[0].split("_")
                    name__ = name[0] + " "
                    if name[1].isalpha():
                        name_ = name[1]
                    else:
                        name_ = ""
                    name__ += name_
                    print(name__)
                    image = Required_Features(image, name__)

                    if image is None:
                        print("No Image or there is an error reading")
                        continue
                    else:
                        image = cv2.resize(image, (140, 140))
                        Gray_Image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        hog_for_image = hog(Gray_Image, block_norm='L2-Hys', feature_vector=True)

                        data_.append(list(hog_for_image))
                        label.append(1)
                        ind += 1
                        print("Ind: {}".format(ind))
                        break

    return data_, label, ind


def Data_Preparation_For_Photos_Without_Face(i):

    for root, directory, filenames in os.walk(path_back):
        for file in filenames:
            path_image = os.path.join(".", root, file)
            image = cv2.imread(path_image)
            image = cv2.resize(image, (140, 140))
            if image is None:
                print("No Image or there is an error reading")
                continue
            else:
                Gray_Image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # print(Gray_Image.shape)
                hog_for_image = hog(Gray_Image, block_norm='L2-Hys', feature_vector=True)
                # print(len(hog_for_image))
                if len(hog_for_image):
                    print("Ind: {}".format(i))
                    data_set.append(list(hog_for_image))
                    labeled.append(0)
                    i += 1


data_set, labeled, index = Data_Preparation_For_Photos_With_Face()
Data_Preparation_For_Photos_Without_Face(index)
print(len(data_set))
print(len(labeled))

# Splitting Data For Test and Train
print(" Splitting Data ")
X_train, X_test, y_train, y_test = train_test_split(data_set, labeled, test_size=0.20, random_state=0)
#
# SVM Classifier
model = SVC(kernel='linear')
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(classification_report(y_test, predictions))
print(accuracy_score(y_test, predictions))

Pkl_Filename = "Face_detector.pkl"

with open(Pkl_Filename, 'wb') as file:
    pickle.dump(model, file)
