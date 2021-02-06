import os
import numpy as np
import cv2

data_path = 'train_data_2'


labels = os.listdir(data_path)
category = np.arange(len(labels))
category_dict = dict(zip(labels, category))
# or labels = ['barack obama', 'donald trump', 'george h bush']
# but we cant give this type of labels bcz for multiple name we cant write like this it will be lengthy
print('Labels:-', labels)  # op ['barack obama', 'donald trump', 'george h bush']
print('Category:-', category)  # op ['barack obama', 'donald trump', 'george h bush']
print('Category Dict:-', category_dict)  # op ['barack obama', 'donald trump', 'george h bush']
# now we have to convert this in to the integers(0, 1, 2....) bcz we cant directly it into string
# means we cant use strings as labels
data = []
target = []
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
print("--------------------------------------------")
for label in labels:
    # print(label)
    imgs_path = os.path.join(data_path, label)  # for path new
    img_names = os.listdir(imgs_path)  # uploading only the img names
    # print(img_names)
    # print('----------------')
    # now we have to go through every images so
    for img_name in img_names:
        img_path = os.path.join(imgs_path, img_name)
        # print(img_path)
        # now load it
        img = cv2.imread(img_path)  # its load an img into an array
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # using color feature in cv2, converting/from BGR to GRAY
        faces = face_classifier.detectMultiScale(gray)
        for (x, y, w, h) in faces:
            cropped_face = gray[y:y+h, x:x+w]
            # cv2.rectangle(img, (x, y), (x+w, y+h), (75, 0, 130), 2)

            cv2.imshow('LIVE:-', cropped_face)
            key = cv2.waitKey(0)  # wait 0 mean waiting forever/infinite
            # asking for user that face will taken or not
            if key == 121:  # ascii value of 121 is y, so we have to press y for accept the image
                cropped_face = cv2.resize(cropped_face, (50, 50))
                data.append(cropped_face)
                target.append(category_dict[label])
                print("THIS IMAGE IS ACCEPTED!!")
            else:
                pass


        '''cv2.imshow('LIVE:-', img)
        cv2.imshow('GRAYSCALE IMAGE:-', gray)
        key = cv2.waitKey(250)
        # waitkey() will wait for 100ms until the key is pressed means Key
        if key == 27:  # 27 is ascii for esc
            break'''


cv2.destroyAllWindows()
# now saving this file into physical file using np.save()
data = np.array(data)
target = np.array(target)

noImages, height, width = data.shape

data = data.reshape(noImages, height*width)  # flatting it for eg. (400,50*50> 400,2500)
np.save('DATA', data)
np.save('TARGET', target)



