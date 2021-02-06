import joblib
import os
import cv2

model = joblib.load('SVM-Face-Recognition.sav')

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

category_dict = {0: 'captim america', 1: 'iron man', 2: 'thor'}
test_data_path = 'test_data'
test_img_names = os.listdir(test_data_path)
print(test_img_names)


for test_img in test_img_names:
    img_path = os.path.join(test_data_path, test_img)
    test_img = cv2.imread(img_path)
    gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)
    for (x, y, w, h) in faces:
        cropped_face = gray[y:y + h, x:x + w]
        # cv2.rectangle(img, (x, y), (x+w, y+h), (75, 0, 130), 2)
        cropped_face = cv2.resize(cropped_face, (50, 50))
        cropped_face = cropped_face.reshape(1, 2500)
        res = model.predict(cropped_face)[0]
        name = category_dict[res]

        cv2.rectangle(test_img, (x, y), (x+w, y+h), (75, 0, 130), 2)
        cv2.putText(test_img, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (75, 0, 130), 2)
    cv2.imshow('LIVE:-', test_img)
    key = cv2.waitKey(1000)
    if key == 27:
        break
cv2.destroyAllWindows()
