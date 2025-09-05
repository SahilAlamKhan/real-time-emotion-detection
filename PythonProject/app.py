from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import os
from keras.models import load_model
from keras.preprocessing.image import img_to_array

# ----- Flask App Init -----
app = Flask(__name__)

# ----- Model and Classifier Load -----
FACE_DETECT_PATH = 'haarcascade_frontalface_default.xml'
MODEL_PATH = 'model.h5'

face_classifier = cv2.CascadeClassifier(FACE_DETECT_PATH)
emotion_classifier = load_model(MODEL_PATH)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# ----- Main Page -----
@app.route("/")
def index():
    return render_template('index.html')

# ----- Real-Time Prediction Route -----
@app.route("/predict", methods=["POST"])
def predict():
    # Get frame from POST request (as file)
    if 'frame' not in request.files:
        return jsonify({'faces': []})

    file = request.files['frame']
    file_bytes = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_rects = face_classifier.detectMultiScale(gray, 1.3, 5)

    faces = []
    for (x, y, w, h) in faces_rects:
        roi_gray = gray[y:y+h, x:x+w]
        try:
            roi_gray_resized = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        except Exception:
            continue  # skip faces that can't be resized

        roi = roi_gray_resized.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        preds = emotion_classifier.predict(roi)[0]
        label = emotion_labels[preds.argmax()]
        faces.append({
            "box": [int(x), int(y), int(w), int(h)],
            "emotion": label,
            "confidence": float(np.max(preds))
        })

    return jsonify({'faces': faces})

# ----- Run the App -----
if __name__ == "__main__":
    # Ensure template and static folders are present for Flask
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
