import cv2
import pickle
import face_recognition
from Testing_Face_detector import Image_Sliding

filename = "Face_Detector.pkl"
model = pickle.load(open(filename, 'rb'))

filename = "Face_Recognizer_Model.pkl"
classifier = pickle.load(open(filename, 'rb'))


capture = cv2.VideoCapture('test.mp4')

while 1:
    _, image = capture.read()
    print(image.shape)
    image = cv2.resize(image, (300, 300))
    rectangles = Image_Sliding(image)
    try:
        print(rectangles)
        encodings = face_recognition.face_encodings(image, rectangles)
        prediction = classifier.predict([encodings[0].tolist()])
        # prediction = prediction[0]
        # cv2.putText(image, text=prediction, org=(rectangles[0][0], rectangles[0][3]),
        #             fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 0, 0), thickness=1)
        cv2.imshow("", image)
        cv2.waitKey(2)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    except:
        cv2.imshow("", image)
        cv2.waitKey(2)
        print("Not Able To encode")

capture.release()
cv2.destroyAllWindows()