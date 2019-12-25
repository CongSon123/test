import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image

#load model
model = model_from_json(open("fer.json", "r").read())
#load weights
model.load_weights('fer.h5')


face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# label
names = ['anger','contempt','disgust','fear','happy','sadness','surprise']

def getLabel(id):
    return ['anger','contempt','disgust','fear','happy','sadness','surprise'][id]

cap=cv2.VideoCapture(0)

while True:
    _, test_img= cap.read()# captures frame and returns boolean value and captured image
    # if not ret:
    #     continue
    # gray_img= cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    faces_detected = face_haar_cascade.detectMultiScale(test_img, 1.32, 7) #scaleFactor=1.32,scaleFactor    Parameter specifying how much the image size is reduced at each image scale.
                                                                            #minNeighbors    Parameter specifying how many neighbors each candidate rectangle should have to retain it.


    for (x,y,w,h) in faces_detected:
        cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=1)
        roi_gray=test_img[y:y+w,x:x+h]#cropping region of interest i.e. face area from  image
        roi_gray=cv2.resize(roi_gray,(48,48))
        
        # Convert to np array
        roi_gray = np.expand_dims(roi_gray, axis = 0) #

        img_data = np.array(roi_gray)
        img_data = img_data.astype('float32')

        img_data = img_data/255
        # img_data[0].shape

        res = model.predict_classes(img_data)
        print('---------', res)


        # img_pixels = image.img_to_arr

# ay(roi_gray)
        # # print (img_pixels.shape)
        # img_pixels = np.expand_dims(img_pixels, axis = 0)
        # # print ('-----------------',img_pixels.shape)

        # img_pixels /= 255


    #     predictions = model.predict(img_pixels)
    #     # print (predictions)

    #     #find max indexed array
    #     max_index = np.argmax(predictions[0])
    #     print (max(predictions[0]))

        # emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        # predicted_emotion = emotions[max_index]

        cv2.putText(test_img, getLabel(res[0]),
         (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (1,0,255), 1)

    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('Facial emotion analysis ',resized_img)



    if cv2.waitKey(10) == ord('q'):#wait until 'q' key is pressed
        break

cap.release()
cv2.destroyAllWindows