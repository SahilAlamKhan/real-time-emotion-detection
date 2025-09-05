from keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2
import numpy as np

# Load models only once
face_classifier = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
classifier = load_model('models/model.h5')
emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

def predict_emotion(image_path):
    # Load and preprocess image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)
    results = []
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi = roi_gray.astype('float')/255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        prediction = classifier.predict(roi)[0]
        label = emotion_labels[prediction.argmax()]
        results.append({'label': label, 'coordinates': (x,y,w,h)})
    return results
