import os
import numpy as np
import cv2
from keras.models import load_model

os.environ['KMP_DUPLICATE_LIB_OK']='True'
# Load the Model
# path_to_model = "./emotion_models/inception_v3/ckp_inception_v3_weights-improvement-136-0.98.hdf5"
# path_to_model = "./emotion_models/inception_resnet_v2/inception_resnet_v2_weights-improvement-129-0.98.hdf5"
path_to_model = "./emotion_models/resnet_50/resnet_50_weights-improvement-176-0.97.hdf5"
model = load_model(path_to_model)
model.compile(optimizer='Adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

# Load the webcam
cap = cv2.VideoCapture(0)
while(True):
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_to_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    resize = cv2.resize(gray_to_bgr, (224, 224))
    tensor_resize = np.expand_dims(gray_to_bgr, axis=0)
    prediction = model.predict(tensor_resize)
    print(np.argmax(prediction[0]))

    cv2.imshow("resized", resize)
    if cv2.waitKey(30) & 0xff == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
